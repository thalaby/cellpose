from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class SegmentationTileDataset(Dataset):
    """
    Tiled dataset for segmentation:
    returns (image tile, mask tile) pairs.
    """
    def __init__(self, sa1b_dataset, tile_size=256, dtype=torch.float32):
        self.base = sa1b_dataset
        self.tile_size = tile_size
        self.dtype = dtype
        self.samples = []  # list of (img_idx, y, x)

        # Precompute tile coordinates for all images
        for img_idx in range(len(self.base)):
            img, mask, class_ids = self.base[img_idx]
            img_arr = np.array(img)   # (H, W, C)
            h, w = img_arr.shape[:2]

            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    self.samples.append((img_idx, y, x))

        print(f"SegmentationTileDataset: {len(self.samples)} tiles")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, y, x = self.samples[idx]
        img, mask, class_ids = self.base[img_idx]

        img_arr = np.array(img)     # (H, W, C)
        mask_arr = np.array(mask)   # (H, W) or (H, W, 1)

        H, W = img_arr.shape[:2]
        ts = self.tile_size

        y_end = min(y + ts, H)
        x_end = min(x + ts, W)

        # 1) Crop both image and mask with the same bounds
        img_tile = img_arr[y:y_end, x:x_end, :]               # (h', w', C)
        mask_tile = mask_arr[y:y_end, x:x_end]                # (h', w') or (h', w', 1)

        # 2) Pad both to (ts, ts)
        h_t, w_t = img_tile.shape[:2]

        if h_t < ts or w_t < ts:
            # image padding
            padded_img = np.zeros((ts, ts, img_tile.shape[2]), dtype=img_tile.dtype)
            padded_img[:h_t, :w_t, :] = img_tile
            img_tile = padded_img

            # mask padding
            if mask_tile.ndim == 2:
                padded_mask = np.zeros((ts, ts), dtype=mask_tile.dtype)
                padded_mask[:h_t, :w_t] = mask_tile
            else:
                padded_mask = np.zeros((ts, ts, mask_tile.shape[2]), dtype=mask_tile.dtype)
                padded_mask[:h_t, :w_t, :] = mask_tile
            mask_tile = padded_mask

        # 3) To tensors
        img_tile = img_tile.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tile).permute(2, 0, 1)   # (C, H, W)

        mask_tensor = torch.from_numpy(mask_tile.astype(np.float32))
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)                  # (1, H, W)
        elif mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.permute(2, 0, 1)[:1]          # (1, H, W)

        return {
            "pixel_values": img_tensor.to(self.dtype),
            "labels": mask_tensor.to(self.dtype),
        }


class StudentSegmentationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        feat = self.encoder(x)     # neck output
        logits = self.decoder(feat)
        return logits
    

class DistillationDatasetWrapper(Dataset):
    """Wrapper for SA1BDataset to work with HuggingFace Trainer
    
    Divides images into 256x256 tiles to handle variable image sizes.
    """
    
    def __init__(self, sa1b_dataset, tile_size=256, dtype=torch.float16):
        self.dataset = sa1b_dataset
        self.tile_size = tile_size
        self.dtype = dtype
        self.tiles = []
        
        # Pre-compute all tiles for all images
        self._create_tiles()
    
    def _create_tiles(self):
        """Create 256x256 tiles from all images in the dataset"""
        for idx in range(len(self.dataset)):
            img, mask, class_ids = self.dataset[idx]
            img_array = np.array(img)  # (H, W, C)
            height, width = img_array.shape[:2]
            
            # Extract tiles
            for y in range(0, height, self.tile_size):
                for x in range(0, width, self.tile_size):
                    # Get tile boundaries (with padding if necessary)
                    y_end = min(y + self.tile_size, height)
                    x_end = min(x + self.tile_size, width)
                    
                    tile = img_array[y:y_end, x:x_end, :]
                    
                    # Pad tile if it's smaller than tile_size x tile_size
                    if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                        padded_tile = np.zeros((self.tile_size, self.tile_size, tile.shape[2]), 
                                              dtype=tile.dtype)
                        padded_tile[:tile.shape[0], :tile.shape[1], :] = tile
                        tile = padded_tile
                    
                    self.tiles.append(tile)
        
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile = self.tiles[idx]
        
        # Normalize to [0, 1]
        tile = tile.astype(np.float32) / 255.0
        # Convert to tensor and permute to (C, H, W)
        img_tensor = torch.from_numpy(tile).permute(2, 0, 1)
        # Convert to the model's dtype
        img_tensor = img_tensor.to(self.dtype)
        
        return {"pixel_values": img_tensor}
    

class DistillationModel(nn.Module):
    """Student model for knowledge distillation"""

    def __init__(self, student_encoder, teacher_encoder):
        super().__init__()
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        # Freeze teacher encoder
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        # Student forward pass
        student_neck = self.student_encoder(pixel_values)

        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_neck = self.teacher_encoder(pixel_values)

        return {"student_neck": student_neck, "teacher_neck": teacher_neck}

class TransformerWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder.patch_embed(x)

        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.neck(x.permute(0, 3, 1, 2))
        return x


class DistillationTrainer(Trainer):
    """Custom Trainer for knowledge distillation"""

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn or nn.MSELoss()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(inputs["pixel_values"])

        student_neck = outputs["student_neck"]
        teacher_neck = outputs["teacher_neck"]

        # Calculate MSE loss between student and teacher neck outputs
        loss = self.loss_fn(student_neck, teacher_neck)

        return (loss, outputs) if return_outputs else loss