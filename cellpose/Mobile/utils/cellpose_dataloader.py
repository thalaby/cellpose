"""
Cellpose-SAM Training Dataloader

Implements the full augmentation pipeline described in the Cellpose-SAM paper:
- Normalization (1st to 99th percentile)
- Random rotation
- Random flipping
- Random resizing (scale factor log-distributed between 0.25 and 4)
- Random cropping to 256x256
- Grayscale conversion (10% of time)
- Contrast inversion (25% of time)
- Third channel dropping (10% of time)
- Channel permutation
- Brightness jitter (std dev 0.2)
- Contrast jitter (factor between -2 and 2)
"""

import numpy as np
import torch
import cv2
import tifffile

from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from cellpose import dynamics

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CellposeSAMLoader(Dataset):
    """
    Dataset that implements the full Cellpose-SAM augmentation pipeline.

    Expected structure:
        root_dir/
            <prefix1>_im.tif
            <prefix1>_mask.tif
            <prefix2>_im.tif
            <prefix2>_mask.tif
            ...

    Args:
        root_dir: Directory containing images and masks
        img_suffix: Suffix for image files (default: '_im')
        mask_suffix: Suffix for mask files (default: '_mask')
        crop_size: Size to crop images to (default: 256)
        mean_cell_diameter: Mean cell diameter for scale normalization (default: 30)
        scale_range: Tuple of (min_scale, max_scale) for augmentation (default: (0.25, 4.0))
        grayscale_prob: Probability of converting to grayscale (default: 0.1)
        invert_prob: Probability of inverting contrast (default: 0.25)
        drop_third_channel_prob: Probability of dropping third channel (default: 0.1)
        brightness_jitter_std: Standard deviation for brightness jitter (default: 0.2)
        contrast_jitter_range: Range for contrast rescaling (default: (-2, 2))
        enable_cache: Whether to cache full images in memory (default: True)
        dtype: Output tensor dtype (default: torch.float32)
    """

    def __init__(
        self,
        root_dir,
        img_suffix="_im",
        mask_suffix="_mask",
        flow_suffix="_flow",
        crop_size=256,
        mean_cell_diameter=30,
        scale_range=(0.25, 4.0),
        grayscale_prob=0.1,
        invert_prob=0.25,
        drop_third_channel_prob=0.1,
        brightness_jitter_std=0.2,
        contrast_jitter_range=(-2, 2),
        enable_cache=True,
        dtype=torch.float32,
        compute_flows=True,
        name=None
    ):
        self.root_dir = Path(root_dir)
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.flow_suffix = flow_suffix
        self.crop_size = crop_size
        self.mean_cell_diameter = mean_cell_diameter
        self.scale_range = scale_range
        self.grayscale_prob = grayscale_prob
        self.invert_prob = invert_prob
        self.drop_third_channel_prob = drop_third_channel_prob
        self.brightness_jitter_std = brightness_jitter_std
        self.contrast_jitter_range = contrast_jitter_range
        self.enable_cache = enable_cache
        self.dtype = dtype
        self.compute_flows = compute_flows
        if name is None:
            self.name = self.root_dir.stem

        # Cache for images and masks
        self.image_cache = {}
        self.mask_cache = {}
        self.flow_cache = {}
        
        # Find all image/mask pairs (or just images if no masks)
        self.samples = self._find_samples()
        
        # Check if dataset has masks
        self.has_masks = any(sample["mask_path"] is not None for sample in self.samples)

        # Compute diameters for each sample (for scale normalization)
        # Load or compute diameters
        diameter_cache_path = self.root_dir / "diameters_cache.npy"
        if diameter_cache_path.exists():
            print(f"Loading cached diameters from {diameter_cache_path}")
            self.diameters = np.load(diameter_cache_path)
            # Verify cache matches number of samples
            if len(self.diameters) != len(self.samples):
                print(
                    f"Warning: Cached diameters ({len(self.diameters)}) doesn't match samples ({len(self.samples)}). Recomputing..."
                )
                self.diameters = self._compute_diameters()
                np.save(diameter_cache_path, self.diameters)
        else:
            print(f"Computing diameters for {len(self.samples)} samples...")
            self.diameters = self._compute_diameters()
            np.save(diameter_cache_path, self.diameters)
            print(f"Saved diameter cache to {diameter_cache_path}")

        print(
            f"CelposeSAMDataset: Found {len(self.samples)} {'image/mask pairs' if self.has_masks else 'images (no masks)'} in {root_dir}"
        )
        print(f"  Crop size: {crop_size}x{crop_size}")
        print(f"  Scale range: {scale_range}")
        print(f"  Mean cell diameter: {mean_cell_diameter}")
        print(f"  Caching: {'enabled' if enable_cache else 'disabled'}")
        print(f"  Has masks: {self.has_masks}")

    def _find_samples(self):
        """Find all image/mask/flow triplets in the directory."""
        samples = []
        img_extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

        # Find all image files
        for ext in img_extensions:
            img_files = list(self.root_dir.glob(f"*{self.img_suffix}{ext}"))

            for img_path in img_files:
                # Get prefix (everything before img_suffix)
                prefix = img_path.stem.replace(self.img_suffix, "")

                # Look for corresponding mask
                mask_path = None
                for mask_ext in img_extensions:
                    candidate = self.root_dir / f"{prefix}{self.mask_suffix}{mask_ext}"
                    if candidate.exists():
                        mask_path = candidate
                        break

                # Look for corresponding flow file
                flow_path = None
                for flow_ext in [".tif", ".tiff"]:  # Flows are typically TIFF
                    candidate = self.root_dir / f"{prefix}{self.flow_suffix}{flow_ext}"
                    if candidate.exists():
                        flow_path = candidate
                        break

                # Add sample even if no mask/flow is found
                samples.append(
                    {
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "flow_path": flow_path,
                        "prefix": prefix,
                    }
                )

        return sorted(samples, key=lambda x: x["prefix"])

    def _compute_diameters(self):
        """Compute diameter for each mask for scale normalization."""
        diameters = []

        for sample in self.samples:
            if sample["mask_path"] is not None:
                mask = self._load_mask(sample["mask_path"])
                diameter = self._compute_diameter(mask)
            else:
                # Use mean cell diameter if no mask available
                diameter = self.mean_cell_diameter
            diameters.append(diameter)

        return np.array(diameters)

    def _compute_diameter(self, mask):
        """Compute mean diameter from a mask."""
        from cellpose import utils

        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask[:, :, 0] if mask.shape[2] > 0 else mask.squeeze()

        diameter, _ = utils.diameters(mask)
        return max(5.0, diameter)  # Minimum diameter of 5

    def _load_image(self, path):
        """Load image with caching."""
        if self.enable_cache and path in self.image_cache:
            return self.image_cache[path].copy()

        try:
            img = tifffile.imread(str(path))
        except Exception:
            img = np.array(Image.open(path))

        # Ensure image is (H, W, C) format
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))

        if self.enable_cache:
            self.image_cache[path] = img.copy()

        return img

    def _load_mask(self, path):
        """Load mask with caching."""
        if self.enable_cache and path in self.mask_cache:
            return self.mask_cache[path].copy()

        try:
            mask = tifffile.imread(str(path))
        except Exception:
            mask = np.array(Image.open(path))

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[2] > 1 else mask.squeeze(-1)

        if self.enable_cache:
            self.mask_cache[path] = mask.copy()

        return mask

    def _load_flow(self, path):
        """Load pre-computed flow with caching."""
        if self.enable_cache and path in self.flow_cache:
            return self.flow_cache[path].copy()

        try:
            flow = tifffile.imread(str(path))
        except Exception:
            flow = np.array(Image.open(path))

        # Flow should be (2, H, W) format: [flowY, flowX]
        if flow.ndim == 3 and flow.shape[0] != 2:
            # If it's (H, W, 2), transpose to (2, H, W)
            if flow.shape[2] == 2:
                flow = np.transpose(flow, (2, 0, 1))

        if self.enable_cache:
            self.flow_cache[path] = flow.copy()

        return flow

    def _normalize_percentile(self, img):
        """Normalize image so 0=1st percentile, 1=99th percentile."""
        img = img.astype(np.float32)

        for c in range(img.shape[-1]):
            channel = img[:, :, c]
            p1 = np.percentile(channel, 1)
            p99 = np.percentile(channel, 99)

            if p99 - p1 > 1e-3:
                img[:, :, c] = (channel - p1) / (p99 - p1)
            else:
                img[:, :, c] = 0

        return img

    def _random_rotate_and_resize(self, img, mask, diameter, flow=None):
        """Apply random rotation, flipping, and resizing to image, mask, and optionally flow."""
        H, W, C = img.shape

        # Random flip
        do_flip = np.random.rand() > 0.5
        if do_flip:
            img = img[:, ::-1, :]
            mask = mask[:, ::-1]
            if flow is not None:
                flow = flow[:, :, ::-1]
                # When flipping horizontally, negate flowX (channel 1)
                flow[1] = -flow[1]

        # Random rotation angle
        theta = np.random.rand() * 360

        # Random scale (log-distributed between scale_range)
        log_scale = np.random.uniform(
            np.log(self.scale_range[0]), np.log(self.scale_range[1])
        )
        scale = np.exp(log_scale)
        scale = np.clip(scale, 0.25, 4.0)

        # Adjust scale based on diameter
        scale *= self.mean_cell_diameter / diameter

        # Calculate new dimensions
        new_H = min(int(H * scale), 20480)
        new_W = min(int(W * scale), 20480)

        # Rotation center
        center = (W / 2, H / 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, theta, scale)

        # Transform image
        img_transformed = cv2.warpAffine(
            img,
            M,
            (new_W, new_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Transform mask
        mask_transformed = cv2.warpAffine(
            mask,
            M,
            (new_W, new_H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Transform flow if provided
        if flow is not None:
            # Flow is (2, H, W) -> transpose to (H, W, 2) for cv2
            flow_hw2 = np.transpose(flow, (1, 2, 0))
            
            # Apply same spatial transformation
            flow_transformed = cv2.warpAffine(
                flow_hw2,
                M,
                (new_W, new_H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            
            # Rotate flow vectors by theta
            # Convert angle to radians
            theta_rad = np.deg2rad(theta)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            
            # Extract flowY and flowX
            flow_y = flow_transformed[:, :, 0]
            flow_x = flow_transformed[:, :, 1]
            
            # Rotate vectors: [flowY', flowX'] = R * [flowY, flowX]
            flow_y_rot = cos_theta * flow_y - sin_theta * flow_x
            flow_x_rot = sin_theta * flow_y + cos_theta * flow_x
            
            # Stack back and transpose to (2, H, W)
            flow_transformed = np.stack([flow_y_rot, flow_x_rot], axis=0)
        else:
            flow_transformed = None

        return img_transformed, mask_transformed, flow_transformed

    def _random_crop(self, img, mask, flow=None):
        """Randomly crop to crop_size x crop_size."""
        H, W, C = img.shape

        # Determine crop coordinates
        y_start = 0
        x_start = 0

        # Handle height
        if H > self.crop_size:
            y_start = np.random.randint(0, H - self.crop_size + 1)
            img = img[y_start : y_start + self.crop_size, :, :]
            mask = mask[y_start : y_start + self.crop_size, :]
            if flow is not None:
                flow = flow[:, y_start : y_start + self.crop_size, :]
            H = self.crop_size

        # Handle width
        if W > self.crop_size:
            x_start = np.random.randint(0, W - self.crop_size + 1)
            img = img[:, x_start : x_start + self.crop_size, :]
            mask = mask[:, x_start : x_start + self.crop_size]
            if flow is not None:
                flow = flow[:, :, x_start : x_start + self.crop_size]
            W = self.crop_size

        # Pad if needed
        if H < self.crop_size or W < self.crop_size:
            img_padded = np.zeros((self.crop_size, self.crop_size, C), dtype=np.float32)
            mask_padded = np.zeros((self.crop_size, self.crop_size), dtype=mask.dtype)

            # Center the content in the padded space
            start_y = (self.crop_size - H) // 2
            start_x = (self.crop_size - W) // 2

            img_padded[start_y : start_y + H, start_x : start_x + W, :] = img
            mask_padded[start_y : start_y + H, start_x : start_x + W] = mask

            img = img_padded
            mask = mask_padded

            if flow is not None:
                flow_padded = np.zeros((2, self.crop_size, self.crop_size), dtype=flow.dtype)
                flow_padded[:, start_y : start_y + H, start_x : start_x + W] = flow
                flow = flow_padded

        return img, mask, flow

    def _convert_to_grayscale(self, img, image_type="auto"):
        """
        Convert image to grayscale based on image type.

        Args:
            img: Image array (H, W, C)
            image_type: 'single_channel', 'h_and_e', 'nuclei', or 'auto'
        """
        H, W, C = img.shape

        # Auto-detect image type
        if image_type == "auto":
            # Simple heuristic: if one channel is all zeros, it's likely nuclei
            zero_channels = [np.all(img[:, :, c] == 0) for c in range(C)]
            if any(zero_channels):
                image_type = "nuclei"
            else:
                image_type = "h_and_e"  # Default

        if image_type == "single_channel":
            # Replicate non-zero channel across all channels
            non_zero_c = None
            for c in range(C):
                if not np.all(img[:, :, c] == 0):
                    non_zero_c = c
                    break
            if non_zero_c is not None:
                gray = img[:, :, non_zero_c : non_zero_c + 1]
                img_gray = np.repeat(gray, 3, axis=2)
            else:
                img_gray = img  # All zeros, keep as is

        elif image_type == "h_and_e":
            # Take mean across all channels
            gray = img.mean(axis=2, keepdims=True)
            img_gray = np.repeat(gray, 3, axis=2)

        elif image_type == "nuclei":
            # Discard nucleus channel, replicate primary channel
            # Assume last channel is nucleus
            primary = img[:, :, 0:1]  # Use first channel as primary
            img_gray = np.repeat(primary, 3, axis=2)

        else:
            img_gray = img

        return img_gray

    def _augment_channels(self, img):
        """
        Apply channel augmentations:
        - Drop third channel (10% probability)
        - Permute channels
        - Brightness jitter per channel
        - Contrast jitter per channel
        """
        H, W, C = img.shape

        # Drop third channel with 10% probability
        if C == 3 and np.random.rand() < self.drop_third_channel_prob:
            img[:, :, 2] = 0

        # Randomly permute channels
        if C == 3:
            perm = np.random.permutation(3)
            img = img[:, :, perm]

        # Brightness jitter (per channel)
        for c in range(C):
            brightness_offset = np.random.normal(0, self.brightness_jitter_std)
            img[:, :, c] = img[:, :, c] + brightness_offset

        # Contrast jitter (per channel)
        for c in range(C):
            contrast_factor = np.random.uniform(
                self.contrast_jitter_range[0], self.contrast_jitter_range[1]
            )
            mean = img[:, :, c].mean()
            img[:, :, c] = (img[:, :, c] - mean) * (2**contrast_factor) + mean

        # Clip to [0, 1]
        img = np.clip(img, 0, 1)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
            - pixel_values: torch.Tensor of shape (3, H, W) normalized to [0, 1]
            - labels: torch.Tensor of shape (3, H, W) with [cellprob, flowY, flowX]
        """

        sample = self.samples[idx]
        diameter = self.diameters[idx]
        # Load image and mask
        img = self._load_image(sample["img_path"])
        
        # Load mask if available, otherwise create zero mask
        if sample["mask_path"] is not None:
            mask = self._load_mask(sample["mask_path"])
        else:
            # Create zero mask with same spatial dimensions as image
            H, W = img.shape[:2]
            mask = np.zeros((H, W), dtype=np.int32)

        # Ensure 3 channels
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 2:
            img = np.concatenate([img, np.zeros_like(img[:, :, :1])], axis=-1)
        elif img.shape[-1] > 3:
            img = img[:, :, :3]

        # Load pre-computed flow if available (and compute_flows is True)
        flow = None
        if self.compute_flows and sample["flow_path"] is not None:
            flow = self._load_flow(sample["flow_path"])  # (2, H, W)

        # Normalize to percentiles BEFORE augmentation
        img = self._normalize_percentile(img)

        # Random rotation, flipping, and resizing (apply to img, mask, and flow)
        img, mask, flow = self._random_rotate_and_resize(img, mask, diameter, flow)

        # Random crop to 256x256 (apply to img, mask, and flow)
        img, mask, flow = self._random_crop(img, mask, flow)

        # Convert to grayscale with 10% probability
        if np.random.rand() < self.grayscale_prob:
            img = self._convert_to_grayscale(img, image_type="auto")

        # Invert contrast with 25% probability
        if np.random.rand() < self.invert_prob:
            img = 1 - img

        # Apply channel augmentations
        img = self._augment_channels(img)

        # Prepare labels
        if self.compute_flows:
            # Compute cell probability
            cellprob = (mask > 0).astype(np.float32)

            if flow is not None:
                # Use pre-computed flow
                flow_y = flow[0]  # (H, W)
                flow_x = flow[1]  # (H, W)
            else:
                # Fallback: compute flows on-the-fly
                flows = dynamics.labels_to_flows(mask[np.newaxis, :, :])[0]  # (3, H, W)
                flow_y = flows[1]
                flow_x = flows[2]

            # Stack: [cellprob, flowY, flowX]
            labels = np.stack([cellprob, flow_y, flow_x], axis=0)  # (3, H, W)
        else:
            # Just use mask as single channel
            labels = mask[np.newaxis, :, :]  # (1, H, W)

        # Convert to torch tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(self.dtype)  # (3, H, W)
        labels_tensor = torch.from_numpy(labels).to(self.dtype)  # (3, H, W) or (1, H, W)

        return {"pixel_values": img_tensor, "labels": labels_tensor}

    def clear_cache(self):
        """Clear all caches to free memory."""
        self.image_cache.clear()
        self.mask_cache.clear()
        self.flow_cache.clear()

    def enable_image_caching(self, enable=True):
        self.enable_cache = enable


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched tensors
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {"pixel_values": pixel_values, "labels": labels}
