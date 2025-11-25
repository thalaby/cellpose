from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from cellpose import models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import os
import re

class ImageMaskDataset(Dataset):
    """
    Loads image/mask pairs from a folder.
    Expected format:
        000_img.png
        000_masks.png
        001_img.png
        001_masks.png
        ...
    """

    def __init__(self, root_dir, img_suffix="_img.png", mask_suffix="_masks.png", dtype=torch.float32):
        self.root_dir = root_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.dtype = dtype

        # Discover all prefixes (e.g., "000")
        files = os.listdir(root_dir)

        # Extract prefixes like "000" from names containing "_img.png"
        self.prefixes = sorted([
            re.match(r"(.*)" + re.escape(img_suffix), f).group(1)
            for f in files
            if f.endswith(img_suffix)
        ])

        if len(self.prefixes) == 0:
            raise RuntimeError(f"No {img_suffix} files found in directory: {root_dir}")

        print(f"Found {len(self.prefixes)} samples in {root_dir}")

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]

        img_path = os.path.join(self.root_dir, prefix + self.img_suffix)
        mask_path = os.path.join(self.root_dir, prefix + self.mask_suffix)

        # Load images
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert to numpy
        img = np.array(img)           # (H, W, 3)
        mask = np.array(mask)         # (H, W)

        # Convert to torch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.to(self.dtype)

        # Mask shape -> (1, H, W)
        mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.permute(2, 0, 1)[:1]

        mask = mask.to(self.dtype)

        return img, mask


class StudentSegmentationModel(nn.Module):
    def __init__(self, encoder, decoder, device="cuda", dtype=torch.float32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dtype = dtype
        self.dummy = torch.zeros((1, 256), device=device, dtype=dtype)

    def forward(self, x):
        feat = self.encoder(x)     # neck output
        out = self.decoder(feat)
        return out, self.dummy

class CellposeCustomModel(models.CellposeModel):
    def __init__(self, gpu=True, nchan=3, custom_net=None, use_bfloat16=False):
        super().__init__(gpu=gpu, nchan=nchan, use_bfloat16=use_bfloat16)
        self.net = custom_net if custom_net is not None else self.net

    

class DistillationDatasetWrapper(Dataset):
    """
    Wrapper that tiles one or multiple datasets into fixed-size image tiles.

    - Accepts:
        * a single Dataset, or
        * a list/tuple of Datasets (e.g. [sa1b_dataset, img_mask_dataset1, img_mask_dataset2, ...])
    - Each underlying dataset item may be:
        * img
        * (img, mask)
        * (img, mask, class_ids)
        * {"pixel_values": img_tensor, ...}
    - This wrapper only uses the image and ignores masks/labels.
    """

    def __init__(self, datasets, tile_size=256, dtype=torch.float16):
        # Allow passing a single dataset or a list of datasets
        if not isinstance(datasets, (list, tuple)):
            datasets = [datasets]

        self.datasets = datasets
        self.tile_size = tile_size
        self.dtype = dtype
        self.tiles = []  # will store numpy arrays of shape (tile_size, tile_size, C)

        self._create_tiles()

    def _extract_image_from_sample(self, sample):
        """
        Given a sample from an arbitrary dataset, extract the image component.

        Supported formats:
            - img
            - (img, mask)
            - (img, mask, class_ids)
            - {"pixel_values": img_tensor, ...}
        """
        # Dict sample (e.g. HuggingFace-like)
        if isinstance(sample, dict):
            if "pixel_values" in sample:
                img = sample["pixel_values"]
            else:
                # Fallback: take first value
                img = next(iter(sample.values()))
        # Tuple sample (img, mask, ...)
        elif isinstance(sample, (list, tuple)):
            img = sample[0]
        else:
            # Just an image
            img = sample

        return img

    def _to_hwc_numpy(self, img):
        """
        Convert an image to HWC numpy array.
        Supports:
            - PIL.Image
            - torch.Tensor in CHW or HWC
            - np.ndarray in CHW or HWC
        """
        # PIL Image
        if isinstance(img, Image.Image):
            arr = np.array(img)
            if arr.ndim == 2:  # grayscale
                arr = arr[..., None]
            return arr  # HWC

        # Torch tensor
        if torch.is_tensor(img):
            arr = img.detach().cpu().numpy()
            # (C, H, W) -> (H, W, C)
            if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.ndim == 2:
                arr = arr[..., None]
            return arr

        # Numpy array
        arr = np.array(img)
        # (C, H, W) -> (H, W, C)
        if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = arr[..., None]
        return arr

    def _create_tiles(self):
        """Create tiles from all images in all datasets."""
        tile_size = self.tile_size

        for ds_idx, ds in enumerate(self.datasets):
            for idx in range(len(ds)):
                sample = ds[idx]
                img = self._extract_image_from_sample(sample)
                img_array = self._to_hwc_numpy(img)  # (H, W, C)

                H, W = img_array.shape[:2]

                for y in range(0, H, tile_size):
                    for x in range(0, W, tile_size):
                        y_end = min(y + tile_size, H)
                        x_end = min(x + tile_size, W)

                        tile = img_array[y:y_end, x:x_end, :]

                        # Pad tile if needed
                        h_t, w_t = tile.shape[:2]
                        if h_t < tile_size or w_t < tile_size:
                            padded = np.zeros(
                                (tile_size, tile_size, tile.shape[2]),
                                dtype=tile.dtype
                            )
                            padded[:h_t, :w_t, :] = tile
                            tile = padded

                        self.tiles.append(tile)

        print(f"DistillationDatasetWrapper: created {len(self.tiles)} tiles "
              f"from {len(self.datasets)} dataset(s).")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]

        # Normalize to [0, 1]
        tile = tile.astype(np.float32) / 255.0

        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(tile).permute(2, 0, 1)

        # Cast to desired dtype
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