from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from cellpose import models
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import os
import re
import tifffile


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

class TiffImageDataset(Dataset):
    """
    Dataset that loads TIFF images from a folder using tifffile.

    Args:
        root_dir (str or Path): Directory containing *.tif or *.tiff images.
        transform (callable, optional): Optional transform applied on torch tensor.
    """

    VALID_EXTS = {".tif", ".tiff"}

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        # Collect all TIFF files
        self.image_paths = sorted([
            p for p in self.root_dir.iterdir()
            if p.suffix.lower() in self.VALID_EXTS
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No TIFF images found in directory: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load TIFF image (preserves dtype, supports multipage/multichannel)
        img = tifffile.imread(img_path)

        # Convert numpy → torch
        img_tensor = torch.from_numpy(img)

        # If grayscale (H, W), add channel dimension
        if img_tensor.ndim == 2:  
            img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)

        # If (H, W, C), convert to (C, H, W)
        elif img_tensor.ndim == 3:  
            img_tensor = img_tensor.permute(2, 0, 1)

        # If TIFF is (Z, H, W) or (Z, H, W, C) → user decides what to do
        # (but we leave it as-is; you can modify if needed)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, str(img_path)



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

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class TiledImageDirDataset(Dataset):
    """
    Dataset for a single image directory that has a precomputed tile_index.npy.

    Directory structure example:
        root_dir/
            img_0001.tif
            img_0002.png
            ...
            tile_index.npy

    tile_index.npy format (per-directory):
        Shape: (N, 6) or (N, 5)
        Columns:
          - if 6: (ds_idx, sample_idx, y, x, H, W)  # ds_idx is ignored
          - if 5: (sample_idx, y, x, H, W)

    Arguments:
        root_dir (str or Path): directory containing images + tile_index.npy
        tile_size (int): tile edge size (B) — tiles are B x B
        dtype (torch.dtype): output tensor dtype
        tile_index_filename (str): name of the .npy file with tile index
        recursive (bool): whether to search for images recursively
    """

    def __init__(
        self,
        root_dir,
        tile_size=256,
        dtype=torch.float16,
        tile_index_filename="tile_index.npy",
        recursive=False,
    ):
        self.root_dir = Path(root_dir)
        self.tile_size = tile_size
        self.dtype = dtype
        self.recursive = recursive

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        # 1) Load tile_index for this directory
        tile_index_path = self.root_dir / tile_index_filename
        if not tile_index_path.is_file():
            raise FileNotFoundError(f"tile_index file not found: {tile_index_path}")

        tile_index = np.load(tile_index_path)
        tile_index = np.asarray(tile_index, dtype=np.int64)

        if tile_index.ndim != 2 or tile_index.shape[1] not in (5, 6):
            raise ValueError(
                f"tile_index at {tile_index_path} must have shape (N,5) or (N,6), "
                f"got {tile_index.shape}"
            )

        # Drop ds_idx column if present
        if tile_index.shape[1] == 6:
            tile_index = tile_index[:, 1:]  # (sample_idx, y, x, H, W)

        # Store per-tile info: (sample_idx, y, x, H, W)
        self.tile_index = tile_index

        # 2) Reconstruct the *same* file ordering as used when tile_index was computed
        # In compute_tile_index_dir.py we used: sorted(files) with optional recursion
        if self.recursive:
            self.image_paths = sorted(
                p for p in self.root_dir.rglob("*") if p.is_file()
            )
        else:
            self.image_paths = sorted(
                p for p in self.root_dir.iterdir() if p.is_file()
            )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in directory: {self.root_dir}")

        print(
            f"TiledImageDirDataset: {len(self.tile_index)} tiles from "
            f"{len(self.image_paths)} files in {self.root_dir}"
        )

    # ---------- helpers ----------

    def _load_image(self, sample_idx: int):
        """
        Load the original image corresponding to sample_idx using PIL,
        convert to HWC numpy array with channels last.
        """
        img_path = self.image_paths[sample_idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB") if img.mode not in ("L", "RGB") else img.copy()
            arr = np.array(img)

        # (H, W) -> (H, W, 1)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr  # HWC

    def _image_array_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        HWC numpy -> normalized CHW torch tensor in self.dtype
        """
        arr = arr.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).to(self.dtype)

    # ---------- Dataset API ----------

    def __len__(self):
        return self.tile_index.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            dict with key "pixel_values": tensor(C, tile_size, tile_size)
        """
        sample_idx, y, x, H, W = self.tile_index[idx]
        tile_size = self.tile_size

        # Load original image
        arr = self._load_image(sample_idx)  # (H, W, C) ideally

        # Use stored H, W only as sanity if you want:
        # assert arr.shape[0] == H and arr.shape[1] == W

        # Extract tile
        y_end = min(y + tile_size, arr.shape[0])
        x_end = min(x + tile_size, arr.shape[1])
        tile = arr[y:y_end, x:x_end, :]

        # Pad if necessary
        h, w = tile.shape[:2]
        if h < tile_size or w < tile_size:
            padded = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=tile.dtype)
            padded[:h, :w, :] = tile
            tile = padded

        img_tensor = self._image_array_to_tensor(tile)
        return {"pixel_values": img_tensor}


class DistillationDatasetWrapperIndex(Dataset):
    """
    Wrapper that concatenates multiple TiledImageDirDataset (or any Dataset with tiles)
    into a single dataset for training.

    Each sub-dataset must implement:
        __len__()
        __getitem__(idx) -> {"pixel_values": tensor(C, H, W), ...}

    This wrapper does NOT care about tile_index; it just delegates to sub-datasets.
    """

    def __init__(self, datasets):
        if not isinstance(datasets, (list, tuple)):
            raise TypeError("datasets must be a list or tuple of Dataset objects.")
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided.")

        self.datasets = list(datasets)

        # Precompute prefix sums for fast index mapping
        self.lengths = [len(ds) for ds in self.datasets]
        self.offsets = []
        running = 0
        for L in self.lengths:
            self.offsets.append(running)
            running += L
        self.total_length = running

        print(
            f"DistillationDatasetWrapper: {len(self.datasets)} datasets, "
            f"total tiles = {self.total_length}"
        )

    def __len__(self):
        return self.total_length

    def _locate(self, idx):
        """
        Map global idx -> (ds_idx, local_idx)
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range 0..{self.total_length-1}")

        # Simple linear scan; can be replaced by bisect if many datasets
        for ds_idx, offset in enumerate(self.offsets):
            next_offset = (
                self.offsets[ds_idx + 1]
                if ds_idx + 1 < len(self.offsets)
                else self.total_length
            )
            if offset <= idx < next_offset:
                return ds_idx, idx - offset

        raise RuntimeError(f"Failed to map global index {idx}")

    def __getitem__(self, idx):
        ds_idx, local_idx = self._locate(idx)
        return self.datasets[ds_idx][local_idx]




class DistillationDatasetWrapper(Dataset):
    """
    Wrapper that combines:
      - tiled images from 'unprocessed_datasets'
      - full images from 'processed_datasets'

    - unprocessed_datasets: list of datasets whose images will be tiled into BxB tiles
    - processed_datasets: list of datasets whose images will be used as-is (no tiling)

    Each underlying dataset item may be:
        * img
        * (img, mask)
        * (img, mask, class_ids)
        * {"pixel_values": img_tensor, ...}

    This wrapper only uses the image and ignores masks/labels.
    __getitem__ always returns:
        {"pixel_values": tensor(C, H, W)} in dtype `self.dtype`.
    """

    def __init__(
        self,
        processed_datasets=None,
        unprocessed_datasets=None,
        tile_size=256,
        dtype=torch.float16,
    ):
        # Normalize inputs to lists
        if processed_datasets is None:
            processed_datasets = []
        if unprocessed_datasets is None:
            unprocessed_datasets = []

        if not isinstance(processed_datasets, (list, tuple)):
            processed_datasets = [processed_datasets]
        if not isinstance(unprocessed_datasets, (list, tuple)):
            unprocessed_datasets = [unprocessed_datasets]

        self.processed_datasets = list(processed_datasets)
        self.unprocessed_datasets = list(unprocessed_datasets)
        self.tile_size = tile_size
        self.dtype = dtype

        # Storage for tiled images (each is a numpy array HxWxC)
        self.tiles = []
        # Precompute number of processed samples for indexing
        self.num_processed = sum(len(ds) for ds in self.processed_datasets)

        # Build tiles only from unprocessed_datasets
        self._create_tiles()

    # ---------- helpers ----------

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
        # Tuple/list sample (img, mask, ...)
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

        # Numpy array (or anything convertible)
        arr = np.array(img)
        # (C, H, W) -> (H, W, C)
        if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = arr[..., None]
        return arr

    def _image_array_to_tensor(self, arr):
        """
        Common path: HWC numpy -> normalized CHW torch tensor in self.dtype
        """
        # Normalize to [0,1]
        arr = arr.astype(np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return img_tensor.to(self.dtype)

    # ---------- tiling ----------

    def _create_tiles(self):
        """Create tiles from all images in all unprocessed datasets."""
        tile_size = self.tile_size
        self.tiles = []

        if not self.unprocessed_datasets:
            print("DistillationDatasetWrapper: no unprocessed_datasets to tile.")
            return

        # Progress bar over unprocessed datasets
        for ds_idx, ds in enumerate(
            tqdm(self.unprocessed_datasets, desc="Unprocessed datasets", unit="dataset")
        ):
            # Progress bar over samples in this dataset
            for idx in tqdm(
                range(len(ds)),
                desc=f"Samples in unprocessed dataset {ds_idx}",
                unit="sample",
                leave=False,
            ):
                sample = ds[idx]
                img = self._extract_image_from_sample(sample)
                img_array = self._to_hwc_numpy(img)  # (H, W, C)

                H, W = img_array.shape[:2]

                # Compute total number of tiles for inner tqdm
                total_tiles = (
                    (H + tile_size - 1) // tile_size
                    * (W + tile_size - 1) // tile_size
                )

                # Tile extraction progress bar
                tile_bar = tqdm(
                    total=total_tiles,
                    desc=f"Tiling image {idx} (ds {ds_idx})",
                    unit="tile",
                    leave=False,
                )

                for y in range(0, H, tile_size):
                    for x in range(0, W, tile_size):
                        y_end = min(y + tile_size, H)
                        x_end = min(x + tile_size, W)
                        tile = img_array[y:y_end, x:x_end, :]

                        # Pad tile if necessary
                        h_t, w_t = tile.shape[:2]
                        if h_t < tile_size or w_t < tile_size:
                            padded = np.zeros(
                                (tile_size, tile_size, tile.shape[2]),
                                dtype=tile.dtype,
                            )
                            padded[:h_t, :w_t, :] = tile
                            tile = padded

                        self.tiles.append(tile)
                        tile_bar.update(1)

                tile_bar.close()

        print(
            f"DistillationDatasetWrapper: created {len(self.tiles)} tiles "
            f"from {len(self.unprocessed_datasets)} unprocessed dataset(s)."
        )

    # ---------- Dataset API ----------

    def __len__(self):
        # total = tiled samples + all processed samples
        return len(self.tiles) + self.num_processed

    def __getitem__(self, idx):
        # Case 1: index in tiled images from unprocessed_datasets
        if idx < len(self.tiles):
            tile = self.tiles[idx]  # HWC numpy
            img_tensor = self._image_array_to_tensor(tile)
            return {"pixel_values": img_tensor}

        # Case 2: index belongs to processed_datasets
        remaining = idx - len(self.tiles)

        for ds in self.processed_datasets:
            ds_len = len(ds)
            if remaining < ds_len:
                sample = ds[remaining]
                img = self._extract_image_from_sample(sample)
                img_array = self._to_hwc_numpy(img)
                img_tensor = self._image_array_to_tensor(img_array)
                return {"pixel_values": img_tensor}
            remaining -= ds_len

        # Should never reach here if __len__ is correct
        raise IndexError(f"Index {idx} out of range for DistillationDatasetWrapper.")
    

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