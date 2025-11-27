from settings import (
    CELL_TRAIN_DATASET_PATHS,
    TEST_DATASET_PATHS,
    SA1B_TRAIN_DATASET_PATH,
    TRAINING_ARGS,
)
import os
import re
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageMaskDataset(Dataset):
    """
    Loads image/mask pairs from a folder.
    Expected format:
        000_img.png
        000_masks.png
        001_img.png
        001_masks.png
        ...
    Supports images of type PNG, JPG, and TIFF.
    """

    def __init__(
        self,
        root_dir,
        img_suffix="_img",
        mask_suffix="_masks",
        dtype=torch.float32,
    ):
        self.root_dir = root_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.dtype = dtype

        # Supported image extensions (lowercase)
        self.img_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        files = os.listdir(root_dir)
        # Optional: debug print
        # print("Files in dir:", files)

        prefixes = set()

        for f in files:
            if not os.path.isfile(os.path.join(root_dir, f)):
                continue  # skip dirs etc.

            name, ext = os.path.splitext(f)
            ext = ext.lower()

            # Check if this file looks like an image file with the img_suffix
            if ext in self.img_extensions and name.endswith(self.img_suffix):
                # prefix is everything before the suffix, e.g. "000" in "000_img"
                prefix = name[: -len(self.img_suffix)]
                prefixes.add(prefix)

        self.prefixes = sorted(prefixes)

        if len(self.prefixes) == 0:
            # Helpful debug message
            raise RuntimeError(
                f"No *{self.img_suffix}* files with supported extensions "
                f"({self.img_extensions}) found in directory: {root_dir}\n"
                f"Example files seen: {files[:10]}"
            )

        print(f"Found {len(self.prefixes)} samples in {root_dir}")

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]

        # Find the image and mask paths with supported extensions
        img_path = next(
            os.path.join(self.root_dir, prefix + self.img_suffix + ext)
            for ext in self.img_extensions
            if os.path.exists(os.path.join(self.root_dir, prefix + self.img_suffix + ext))
        )
        mask_path = next(
            os.path.join(self.root_dir, prefix + self.mask_suffix + ext)
            for ext in self.img_extensions
            if os.path.exists(os.path.join(self.root_dir, prefix + self.mask_suffix + ext))
        )

        # Load images
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert to numpy
        img = np.array(img)  # (H, W, 3)
        mask = np.array(mask)  # (H, W)

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
    

def _tile_512_to_256(x):
    """
    x: array with shape (N, 512, 512, C) or (N, 512, 512)
    Returns: (4N, 256, 256, C) or (4N, 256, 256)
    """

    # Add channel dim if missing (for masks)
    added_channel = False
    if x.ndim == 3:
        x = x[..., None]      # (N, 512, 512, 1)
        added_channel = True

    N, H, W, C = x.shape
    assert H == 512 and W == 512, f"Expected 512x512, got {x.shape}"

    # (N, 512, 512, C) → split into quadrants
    # reshape: (N, 2, 256, 2, 256, C)
    x = x.reshape(N, 2, 256, 2, 256, C)

    # transpose: (N, 2, 2, 256, 256, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)

    # merge first 3 dims → (4N, 256, 256, C)
    x = x.reshape(N * 4, 256, 256, C)

    # Remove channel for masks if originally single-channel
    if added_channel:
        x = x[..., 0]  # (4N, 256, 256)

    return x


class NPZImageMaskDataset(Dataset):
    """
    Loads image/mask pairs from NPZ files.

    Expected NPZ format:
        - 'X': images array with shape (N, H, W, C) or (N, C, H, W)
        - 'y': masks array with shape (N, H, W) or (N, H, W, C)

    Compatible with datasets like TissueNet.

    Args:
        npz_path (str or Path): Path to the NPZ file
        dtype (torch.dtype): Output tensor dtype
        name (str, optional): Dataset name for tracking. Defaults to NPZ filename.
    """

    def __init__(self, npz_path, dtype=torch.float32, name=None, bsize=256):
        self.npz_path = Path(npz_path)
        self.dtype = dtype
        self.name = name if name is not None else self.npz_path.stem

        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        # Load NPZ data
        npz_data = np.load(str(self.npz_path))

        if "X" not in npz_data or "y" not in npz_data:
            raise ValueError(
                f"NPZ file must contain 'X' and 'y' keys. Found: {list(npz_data.keys())}"
            )

        self.images = _tile_512_to_256(npz_data["X"])
        self.masks = _tile_512_to_256(npz_data["y"])

        # ------------------------------
        # Normalize images
        # ------------------------------
        

        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Number of images ({len(self.images)}) must match number of masks ({len(self.masks)})"
            )

        print(f"Loaded NPZ dataset '{self.name}' from {npz_path}")
        print(f"  {len(self.images)} samples")
        print(f"  Image shape: {self.images.shape}, dtype: {self.images.dtype}")
        print(f"  Mask shape: {self.masks.shape}, dtype: {self.masks.dtype}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image_tensor, mask_tensor)
                - image_tensor: torch.Tensor of shape (C, H, W)
                - mask_tensor: torch.Tensor of shape (1, H, W)
        """
        img = self.images[idx]  # (H, W, C) or (C, H, W)
        mask = self.masks[idx]  # (H, W) or (H, W, C)

        # Normalize image to [0, 1] if needed
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            # Normalize if values are in [0, 255] range
            if img.max() > 1.0:
                img = img / 255.0

        # Handle different image formats
        if img.ndim == 3:
            # Check if channels are last (H, W, C) or first (C, H, W)
            if img.shape[-1] in [1, 2, 3, 4]:  # Likely (H, W, C)
                img = np.transpose(img, (2, 0, 1))  # -> (C, H, W)
            # else: already (C, H, W)
        elif img.ndim == 2:  # Grayscale (H, W)
            img = img[None, ...]  # -> (1, H, W)

        # Handle mask formats
        if mask.ndim == 3:
            # If mask has channels, take first channel or squeeze
            if mask.shape[-1] == 1:  # (H, W, 1)
                mask = mask[..., 0]
            elif mask.shape[0] == 1:  # (1, H, W)
                mask = mask[0]
            else:  # Take first channel if multiple
                mask = mask[..., 0] if mask.shape[-1] > 1 else mask[0]

        # Ensure mask is 2D
        if mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape after processing: {mask.shape}")

        # Convert to torch tensors
        img_tensor = torch.from_numpy(img).to(self.dtype)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.dtype)  # (1, H, W)

        # Ensure img_tensor has 3 channels
        # img_tensor is now (C, H, W), so check shape[0]
        channels = img_tensor.shape[0]
        if channels < 3:  # Grayscale or 2-channel
            # Pad along channel dimension (dim=0) to make it 3 channels
            padding = torch.zeros(3 - channels, img_tensor.shape[1], img_tensor.shape[2], dtype=img_tensor.dtype, device=img_tensor.device)
            img_tensor = torch.cat([img_tensor, padding], dim=0)
        return {"pixel_values": img_tensor, "labels": mask_tensor}


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
            self.image_paths = sorted(p for p in self.root_dir.iterdir() if p.is_file())

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

        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(self.dtype)
        # Pad the channels dimension to be of size 3 with zeros if necessary
        if tensor.shape[0] < 3:
            padding = torch.zeros(3 - tensor.shape[0], tensor.shape[1], tensor.shape[2], dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor

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
            raise IndexError(f"Index {idx} out of range 0..{self.total_length - 1}")

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


class CombinedImageMaskDataset(Dataset):
    """
    Combines multiple ImageMaskDataset instances from different paths.
    Returns image and mask pairs for segmentation evaluation as dictionaries.
    """

    def __init__(
        self,
        dataset_paths,
        img_suffix="_img",
        mask_suffix="_masks",
        dtype=torch.float32,
    ):
        self.datasets = []
        self.offsets = [0]

        for path in dataset_paths:
            if str(path).endswith(".npz"):
                ds = NPZImageMaskDataset(path, dtype=dtype)
            else:
                ds = ImageMaskDataset(
                    path, img_suffix=img_suffix, mask_suffix=mask_suffix, dtype=dtype
                )
            self.datasets.append(ds)
            self.offsets.append(self.offsets[-1] + len(ds))

        self.total_length = self.offsets[-1]
        print(
            f"CombinedImageMaskDataset: {len(self.datasets)} datasets, total samples = {self.total_length}"
        )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i in range(len(self.datasets)):
            if idx < self.offsets[i + 1]:
                local_idx = idx - self.offsets[i]
                img, mask = self.datasets[i][local_idx]
                # Return as dict for compatibility with Trainer's data collator
                return {"pixel_values": img, "labels": mask}
        raise IndexError(f"Index {idx} out of range")


def get_train_dataset():
    datasets = []
    paths = (
        CELL_TRAIN_DATASET_PATHS
        if TRAINING_ARGS.get("train_on_cellular", True)
        else SA1B_TRAIN_DATASET_PATH
    )
    for path in paths:
        if str(path).endswith(".npz"):
            ds = NPZImageMaskDataset(path, dtype=torch.float32)
        else:
            ds = TiledImageDirDataset(root_dir=path, dtype=torch.float32)
        datasets.append(ds)
    combined_dataset = DistillationDatasetWrapperIndex(datasets=datasets)
    return combined_dataset


def get_test_dataset():
    return CombinedImageMaskDataset(TEST_DATASET_PATHS, dtype=torch.float32)
