import os
import random
import time
import numpy as np
import torch
import logging
import tifffile

from cellpose.Mobile.utils.settings import (
    CELL_TRAIN_DATASET_PATHS,
    TEST_DATASET_PATHS,
    SA1B_TRAIN_DATASET_PATH,
    TRAINING_ARGS,
    CELL_TRAIN_DATASET_PATHS_DECODER,
    SA1B_DATASET_PATH,
)
from cellpose.Mobile.utils.cellpose_dataloader import CellposeSAMLoader
from cellpose.Mobile.utils.sa import SA1BDataset
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from cellpose.Mobile.utils.sa_dataset_utils import TiledSA1BDatasetLazy
from cellpose import dynamics

# Allow PIL to load truncated images instead of raising an error
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class TiledImageMaskDataset(Dataset):
    """
    Dataset that uses tile_index.npy and tile_index_mask_info.npy files
    created by create_tiles.py to load image/mask tiles efficiently.

    Expected structure:
        root_dir/
            <prefix1>_im.tif
            <prefix1>_mask.tif  (optional)
            <prefix2>_im.tif
            <prefix2>_mask.tif  (optional)
            ...
            tile_index.npy  # (N, 7) array: (ds_idx, sample_idx, y, x, H, W, mask_sample_idx)
            tile_index_mask_info.npy  # dict mapping sample_idx -> {'mask_path': str, 'flow_path': str} (optional)

    Args:
        root_dir: Directory containing images, masks, and tile_index files
        dtype: Output tensor dtype (default: torch.float32)
        name: Optional dataset name for logging
        img_suffix: Suffix for image files (default: '_im')
        mask_suffix: Suffix for mask files (default: '_mask')
    """

    def __init__(
        self,
        root_dir,
        dtype=torch.float32,
        name=None,
        img_suffix="_im",
        mask_suffix="_mask",
        enable_cache=False,
    ):
        self.root_dir = Path(root_dir)
        self.dtype = dtype
        self.name = name or self.root_dir.name
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.image_cache = {}
        self.enable_cache = enable_cache

        # Load tile index
        tile_index_path = self.root_dir / "tile_index.npy"
        if not tile_index_path.exists():
            raise FileNotFoundError(
                f"tile_index.npy not found in {root_dir}. "
                f"Please run create_tiles.py first."
            )

        self.tile_index = np.load(tile_index_path)

        # tile_index can have shape (N, 6) or (N, 7)
        # (N, 6): (ds_idx, sample_idx, y, x, H, W) - no masks
        # (N, 7): (ds_idx, sample_idx, y, x, H, W, mask_sample_idx) - with masks
        if self.tile_index.shape[1] not in [6, 7]:
            raise ValueError(
                f"Expected tile_index to have 6 or 7 columns, got {self.tile_index.shape[1]}. "
                f"Make sure create_tiles.py was run correctly."
            )

        # Load mask info (optional)
        mask_info_path = self.root_dir / "tile_index_mask_info.npy"
        self.has_masks = mask_info_path.exists()

        if self.has_masks:
            self.mask_info = np.load(mask_info_path, allow_pickle=True).item()
        else:
            self.mask_info = {}
            logger.info(
                f"No mask info found for dataset '{self.name}', will return zero masks"
            )

        # Build list of image files for quick access
        self.image_files = self._build_image_file_list()

        mask_status = "with masks" if self.has_masks else "without masks"
        print(
            f"TiledImageMaskDataset '{self.name}': {len(self)} tiles from {len(self.image_files)} images ({mask_status})"
        )
    
    def enable_image_caching(self, enable=True):
        """Enable or disable image caching."""
        self.enable_cache = enable
        if not enable:
            self.image_cache.clear()

    def _build_image_file_list(self):
        """Build a list of image file paths indexed by sample_idx."""
        # Get unique sample indices from tile_index
        sample_indices = np.unique(self.tile_index[:, 1])

        # Find all image files
        image_files = {}

        if self.has_masks:
            # Match images via mask_info
            for img_file in self.root_dir.glob(f"*{self.img_suffix}.tif*"):
                for sample_idx in sample_indices:
                    if sample_idx in self.mask_info:
                        mask_path = Path(self.mask_info[sample_idx]["mask_path"])
                        # Get prefix from mask path
                        prefix = mask_path.stem.replace(self.mask_suffix, "")
                        expected_img_name = prefix + self.img_suffix + img_file.suffix

                        if img_file.name == expected_img_name:
                            image_files[sample_idx] = img_file
                            break
        else:
            # No masks: directly map images by sample_idx
            for img_file in self.root_dir.glob(f"*{self.img_suffix}.tif*"):
                # Extract prefix from filename
                prefix = img_file.stem.replace(self.img_suffix, "")
                # Try to parse sample_idx from prefix (assuming numeric or similar pattern)
                # For simplicity, we'll enumerate images in sorted order
                pass

            # Enumerate all image files in sorted order
            sorted_img_files = sorted(self.root_dir.glob(f"*{self.img_suffix}.tif*"))
            for idx, img_file in enumerate(sorted_img_files):
                if idx in sample_indices:
                    image_files[idx] = img_file

        # Convert to list indexed by sample_idx
        max_idx = int(sample_indices.max())
        file_list = [None] * (max_idx + 1)
        for sample_idx, path in image_files.items():
            file_list[sample_idx] = path

        return file_list

    def _add_to_cache(self, path, array):
        """Add image array to cache."""
        self.image_cache[path] = array
        if len(self.image_cache) > 100:  # Limit cache size to 100 items
            self.image_cache.clear()

    def __len__(self):
        return len(self.tile_index)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
            - pixel_values: torch.Tensor of shape (3, H, W) normalized to [0, 1]
            - labels: torch.Tensor of shape (1, H, W) with mask values
        """
        # Get tile info: can be (ds_idx, sample_idx, y, x, H, W) or (ds_idx, sample_idx, y, x, H, W, mask_sample_idx)
        tile_info = self.tile_index[idx]
        if len(tile_info) == 7:
            ds_idx, sample_idx, y, x, H, W, mask_sample_idx = tile_info
        else:  # len == 6
            ds_idx, sample_idx, y, x, H, W = tile_info
            mask_sample_idx = -1  # No mask

        sample_idx = int(sample_idx)
        mask_sample_idx = int(mask_sample_idx)
        y, x = int(y), int(x)

        # Load image
        img_path = self.image_files[sample_idx]
        if img_path is None:
            raise RuntimeError(
                f"No image file found for sample_idx={sample_idx} tileinfo={tile_info} image_dir={self.root_dir}"
            )

        # Load full image using tifffile for better TIFF support
        if img_path in self.image_cache:
            img = self.image_cache[img_path]
        else:
            try:
                img = tifffile.imread(str(img_path))
            except Exception:
                # Fallback to PIL
                img = np.array(Image.open(img_path))
            if self.enable_cache:
                self._add_to_cache(img_path, img)

        # Ensure image is (H, W, C) format
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # Add channel dimension
        elif img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W) format
            img = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)

        # Validate tile coordinates
        if y >= img.shape[0] or x >= img.shape[1] or y < 0 or x < 0:
            raise ValueError(
                f"Invalid tile coordinates for idx={idx}: "
                f"y={y}, x={x}, but image shape is {img.shape[:2]} (H, W). "
                f"Image path: {img_path}, "
                f"tile_info: {self.tile_index[idx]}"
            )

        # Extract tile from full image
        tile_h = min(256, img.shape[0] - y)
        tile_w = min(256, img.shape[1] - x)

        # Additional check for negative dimensions
        if tile_h <= 0 or tile_w <= 0:
            raise ValueError(
                f"Calculated negative tile dimensions for idx={idx}: "
                f"tile_h={tile_h}, tile_w={tile_w}. "
                f"Image shape: {img.shape[:2]} (H, W), y={y}, x={x}. "
                f"Image path: {img_path}, "
                f"tile_info: {self.tile_index[idx]}"
            )

        img_tile = img[y : y + tile_h, x : x + tile_w]

        # Load mask if available, otherwise create zero mask
        if self.has_masks and mask_sample_idx != -1:
            mask_path = Path(self.mask_info[mask_sample_idx]["mask_path"])
            try:
                mask = tifffile.imread(str(mask_path))
            except Exception:
                mask = np.array(Image.open(mask_path))
            
            assert mask.shape == img.shape[:2], f"Mask shape {mask.shape} does not match image shape {img.shape[:2]} for mask_path {mask_path}"

            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = mask[:, :, 0] if mask.shape[2] > 1 else mask.squeeze(-1)

            mask_tile = mask[y : y + tile_h, x : x + tile_w]
            
            # Convert to PyTorch-compatible dtype if needed
            # PyTorch doesn't support ulonglong, so convert to a supported type
            if mask_tile.dtype == np.ulonglong or mask_tile.dtype == np.longlong:
                mask_tile = mask_tile.astype(np.int64)
            elif mask_tile.dtype not in [np.float64, np.float32, np.float16, 
                                          np.int64, np.int32, np.int16, np.int8,
                                          np.uint64, np.uint32, np.uint16, np.uint8,
                                          bool]:
                # Fallback: convert unsupported types to float32
                mask_tile = mask_tile.astype(np.float32)
        else:
            # No mask available, create zero mask
            mask_tile = np.zeros((tile_h, tile_w), dtype=np.float32)

        # Pad tiles to 256x256 if needed (for edge tiles)
        if tile_h < 256 or tile_w < 256:
            # Pad image tile
            pad_h = 256 - tile_h
            pad_w = 256 - tile_w
            img_tile_padded = np.zeros(
                (256, 256, img_tile.shape[-1]), dtype=img_tile.dtype
            )
            img_tile_padded[:tile_h, :tile_w] = img_tile
            img_tile = img_tile_padded

            # Pad mask tile
            mask_tile_padded = np.zeros((256, 256), dtype=mask_tile.dtype)
            mask_tile_padded[:tile_h, :tile_w] = mask_tile
            mask_tile = mask_tile_padded

        # Normalize image to [0, 1]
        if img_tile.dtype == np.uint8:
            img_tile = img_tile.astype(np.float32) / 255.0
        elif img_tile.dtype == np.uint16:
            img_tile = img_tile.astype(np.float32) / 65535.0
        else:
            img_tile = img_tile.astype(np.float32)
            if img_tile.max() > 1.0:
                img_tile = img_tile / img_tile.max()

        # Ensure 3 channels
        if img_tile.shape[-1] == 1:
            img_tile = np.repeat(img_tile, 3, axis=-1)
        elif img_tile.shape[-1] == 2:
            # Pad to 3 channels
            img_tile = np.concatenate(
                [img_tile, np.zeros_like(img_tile[:, :, :1])], axis=-1
            )
        elif img_tile.shape[-1] > 3:
            img_tile = img_tile[:, :, :3]

        # Convert to torch tensors
        img_tensor = (
            torch.from_numpy(img_tile).permute(2, 0, 1).to(self.dtype)
        )  # (3, H, W)
        mask_tensor = (
            torch.from_numpy(mask_tile).unsqueeze(0).to(self.dtype)
        )  # (1, H, W)

        return {"pixel_values": img_tensor, "labels": mask_tensor}


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
        img_suffix="_im",
        mask_suffix="_mask",
        dtype=torch.float32,
        name=None,
    ):
        self.root_dir = root_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.dtype = dtype
        self.name = name
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
            if os.path.exists(
                os.path.join(self.root_dir, prefix + self.img_suffix + ext)
            )
        )
        mask_path = next(
            os.path.join(self.root_dir, prefix + self.mask_suffix + ext)
            for ext in self.img_extensions
            if os.path.exists(
                os.path.join(self.root_dir, prefix + self.mask_suffix + ext)
            )
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

        return {"pixel_values": img, "labels": mask}


def _tile_512_to_256(x):
    """
    x: array with shape (N, 512, 512, C) or (N, 512, 512)
    Returns: (4N, 256, 256, C) or (4N, 256, 256)
    """

    # Add channel dim if missing (for masks)
    added_channel = False
    if x.ndim == 3:
        x = x[..., None]  # (N, 512, 512, 1)
        added_channel = True

    N, H, W, C = x.shape
    if H == 256 and W == 256:
        return x
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
        self.name = name

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
            padding = torch.zeros(
                3 - channels,
                img_tensor.shape[1],
                img_tensor.shape[2],
                dtype=img_tensor.dtype,
                device=img_tensor.device,
            )
            img_tensor = torch.cat([img_tensor, padding], dim=0)
        # flows = dynamics.labels_to_flows(mask_tensor)[0]
        return {"pixel_values": img_tensor, "labels": mask_tensor} # , "flows": flows}


class DistillationDatasetWrapperIndex(Dataset):
    """
    Wrapper that concatenates multiple TiledImageDirDataset (or any Dataset with tiles)
    into a single dataset for training.

    Each sub-dataset must implement:
        __len__()
        __getitem__(idx) -> {"pixel_values": tensor(C, H, W), ...}

    This wrapper does NOT care about tile_index; it just delegates to sub-datasets.
    """

    def __init__(self, datasets, C=3, H=256, W=256):
        if not isinstance(datasets, (list, tuple)):
            raise TypeError("datasets must be a list or tuple of Dataset objects.")
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided.")

        self.datasets = list(datasets)
        self.C = C
        self.H = H
        self.W = W

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
        sample = self.datasets[ds_idx][local_idx]

        # Add padding to the image to ensure it is C x H x W
        img = sample["pixel_values"]
        _, h, w = img.shape
        if h < self.H or w < self.W:
            padding = (0, self.W - w, 0, self.H - h)  # (left, right, top, bottom)
            img = torch.nn.functional.pad(img, padding, mode="constant", value=0)

        # Ensure the image has the correct number of channels
        if img.shape[0] < self.C:
            padding_channels = torch.zeros(
                self.C - img.shape[0], img.shape[1], img.shape[2], dtype=img.dtype
            )
            img = torch.cat([img, padding_channels], dim=0)

        sample["pixel_values"] = img
        return sample


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

        for name, path in dataset_paths:
            if str(path).endswith(".npz"):
                ds = NPZImageMaskDataset(path, dtype=dtype, name=name)
            else:
                ds = ImageMaskDataset(
                    path,
                    img_suffix=img_suffix,
                    mask_suffix=mask_suffix,
                    dtype=dtype,
                    name=name,
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


def get_train_val_dataset_distilled():
    train_datasets = []
    val_datasets = []
    paths = (
        CELL_TRAIN_DATASET_PATHS
        if TRAINING_ARGS.get("train_on_cellular", True)
        else SA1B_TRAIN_DATASET_PATH
    )
    for path in paths:
        val_ds = None
        if str(path).endswith(".npz"):
            train_ds = NPZImageMaskDataset(path, dtype=torch.float32)
        else:
            ds = TiledImageDirDataset(root_dir=path, dtype=torch.float32)
            # Split dataset into 90% train and 10% validation
            train_size = int(0.9 * len(ds))
            val_size = len(ds) - train_size
            train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
        train_datasets.append(train_ds)
        if val_ds is not None:
            val_datasets.append(val_ds)
    combined_train_dataset = DistillationDatasetWrapperIndex(datasets=train_datasets)
    val_datasets.extend(get_val_dataset())
    combined_val_dataset = DistillationDatasetWrapperIndex(datasets=val_datasets)
    return combined_train_dataset, combined_val_dataset


def get_test_dataset():
    datasets = {}
    for name, path in TEST_DATASET_PATHS:
        if str(path).endswith(".npz"):
            ds = NPZImageMaskDataset(path, dtype=torch.float32, name=name)
        else:
            ds = ImageMaskDataset(path, dtype=torch.float32, name=name)
        datasets[name] = ds
    return datasets


def get_train_val_dataset_decoder():
    train_datasets = []
    val_datasets = []
    paths = CELL_TRAIN_DATASET_PATHS_DECODER
    for path in paths:
        val_ds = None
        if str(path).endswith(".npz"):
            train_ds = NPZImageMaskDataset(path, dtype=torch.float32)
        else:
            ds = TiledImageDirDataset(
                root_dir=path, dtype=torch.float32, masks_enabled=True
            )
            # Split dataset into 90% train and 10% validation
            train_size = int(0.9 * len(ds))
            val_size = len(ds) - train_size
            train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
        train_datasets.append(train_ds)
        if val_ds is not None:
            val_datasets.append(val_ds)
    combined_train_dataset = DistillationDatasetWrapperIndex(datasets=train_datasets)
    combined_val_dataset = DistillationDatasetWrapperIndex(datasets=val_datasets)
    return combined_train_dataset, combined_val_dataset


def get_train_dataset_sa1b():
    dataset = TiledImageDirDataset(
        root_dir=Path(SA1B_DATASET_PATH, "images"),
        dtype=torch.float32,
        masks_enabled=True,
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    return train_dataset, val_dataset


def get_val_dataset():
    datasets = []
    for path in CELL_EVAL_DATASET_PATHS:
        assert str(path).endswith(".npz")
        ds = NPZImageMaskDataset(path, dtype=torch.float32)
        datasets.append(ds)
    return datasets


def load_dataset_auto(
    path, dtype=torch.float32, name=None, img_suffix="_im", mask_suffix="_mask"
):
    """
    Automatically detect and load the appropriate dataset type based on path contents.

    Priority:
        1. If path ends with .npz -> NPZImageMaskDataset
        2. If path contains tile_index.npy -> TiledImageMaskDataset
        3. Otherwise -> ImageMaskDataset

    Args:
        path: Path to dataset directory or NPZ file
        dtype: Output tensor dtype
        name: Optional dataset name
        img_suffix: Suffix for image files (for TiledImageMaskDataset and ImageMaskDataset)
        mask_suffix: Suffix for mask files

    Returns:
        Dataset instance
    """
    path = Path(path)

    # Check if it's an NPZ file
    if str(path).endswith(".npz"):
        return NPZImageMaskDataset(path, dtype=dtype, name=name or path.stem)

    # Check if it's a directory with tile_index
    if path.is_dir():
        print(f"Loading CellposeSAMLoader from {path}")
        ds = CellposeSAMLoader(
            path,
            dtype=dtype,
            img_suffix=img_suffix,
            mask_suffix=mask_suffix,
            enable_cache=False,
            compute_flows=False,
        )
        t0 = time.time()
        sample_load_timed = ds[random.randint(0, len(ds) - 1)]
        t1 = time.time()
        if t1 - t0 > 0.1:
            ds.enable_image_caching(True)
        return ds

    # Fallback to ImageMaskDataset
    return ImageMaskDataset(
        path,
        img_suffix=img_suffix,
        mask_suffix=mask_suffix,
        dtype=dtype,
        name=name or path.name,
    )


def get_train_val_dataset_formatted():
    """
    Load training and validation datasets from Data/CellDatasets/Formatted.
    Automatically detects whether to use TiledImageMaskDataset or ImageMaskDataset
    based on presence of tile_index.npy files.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    train_datasets = []
    val_datasets = []

    paths = (
        CELL_TRAIN_DATASET_PATHS
        if TRAINING_ARGS.get("train_on_cellular", True)
        else SA1B_TRAIN_DATASET_PATH
    )

    for path in paths:
        path = Path(path)

        # Load dataset with auto-detection
        ds = load_dataset_auto(
            path, dtype=torch.float32, img_suffix="_im", mask_suffix="_mask"
        )

        # If it's an NPZ dataset, no splitting needed (usually pre-split)
        if isinstance(ds, NPZImageMaskDataset):
            train_datasets.append(ds)
        else:
            # Split into 90% train, 10% validation
            train_size = int(0.9 * len(ds))
            val_size = len(ds) - train_size

            if val_size > 0:
                train_ds, val_ds = torch.utils.data.random_split(
                    ds, [train_size, val_size]
                )
                train_datasets.append(train_ds)
                val_datasets.append(val_ds)
            else:
                train_datasets.append(ds)


    # Combine all datasets
    combined_train_dataset = DistillationDatasetWrapperIndex(datasets=train_datasets)
    combined_val_dataset = DistillationDatasetWrapperIndex(datasets=val_datasets)

    return combined_train_dataset, combined_val_dataset
