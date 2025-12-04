"""
SA-1B Dataset with tiling support.

This module provides a dataset class that loads SA-1B images and masks
with tiling functionality similar to TiledImageDirDataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from sa import SA1BDataset

logger = logging.getLogger(__name__)


class TiledSA1BDataset(Dataset):
    """
    SA-1B Dataset with tiling support similar to TiledImageDirDataset.
    
    This dataset loads images and their corresponding masks from SA-1B dataset,
    divides them into tiles of specified size, and returns both image tiles
    and their corresponding mask tiles.
    
    Args:
        dataset_dir (str): Root directory of SA-1B dataset
        tile_size (int): Size of each tile (default: 256)
        dtype (torch.dtype): Output tensor dtype (default: torch.float16)
        ids (list, optional): List of sample IDs to load
        annotation_dir (str): Annotation directory name (default: 'annotations')
        image_dir (str): Image directory name (default: 'images')
        min_object (int): Minimum pixels for valid object (default: 0)
    """
    
    def __init__(
        self,
        dataset_dir,
        tile_size=256,
        dtype=torch.float16,
        ids=None,
        annotation_dir='annotations',
        image_dir='images',
        min_object=0,
    ):
        self.dataset_dir = dataset_dir
        self.tile_size = tile_size
        self.dtype = dtype
        self.min_object = min_object
        
        # Load SA1B dataset
        self.sa1b_dataset = SA1BDataset(
            dataset_dir=dataset_dir,
            ids=ids,
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            min_object=min_object,
        )
        
        # Pre-compute all tiles
        self.tiles = []
        self._create_tiles()
        
        logger.info(
            f"TiledSA1BDataset: {len(self.tiles)} tiles from "
            f"{len(self.sa1b_dataset)} images in {dataset_dir}"
        )
    
    def _create_tiles(self):
        """
        Pre-compute tile information for all images.
        Stores: (image_idx, y, x, y_end, x_end, img_height, img_width)
        """
        logger.info(f"Creating {self.tile_size}x{self.tile_size} tiles from SA-1B dataset...")
        
        for img_idx in range(len(self.sa1b_dataset)):
            # Get image info to determine dimensions
            sample = self.sa1b_dataset[img_idx]
            img = sample["pixel_values"]  # PIL Image
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Create tiles for this image
            for y in range(0, img_height, self.tile_size):
                for x in range(0, img_width, self.tile_size):
                    y_end = min(y + self.tile_size, img_height)
                    x_end = min(x + self.tile_size, img_width)
                    
                    # Store tile info: (image_idx, y, x, y_end, x_end, img_height, img_width)
                    self.tiles.append((img_idx, y, x, y_end, x_end, img_height, img_width))
        
        logger.info(f"Created {len(self.tiles)} tiles")
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary with:
            - pixel_values: image tile tensor (C, tile_size, tile_size)
            - labels: mask tile tensor (tile_size, tile_size) with instance IDs
            - class_ids: array of class IDs present in this tile
        """
        img_idx, y, x, y_end, x_end, img_height, img_width = self.tiles[idx]
        
        # Load the full image and mask from SA1B dataset
        sample = self.sa1b_dataset[img_idx]
        img = sample["pixel_values"]  # PIL Image
        mask = sample["labels"]  # (H, W, num_instances)
        class_ids = sample["class_ids"]  # array of class IDs
        
        # Convert PIL image to numpy array
        img_array = np.array(img)  # (H, W, C)
        
        # Extract tile from image
        img_tile = img_array[y:y_end, x:x_end, :]
        
        # Extract tile from mask
        mask_tile = mask[y:y_end, x:x_end, :]  # (tile_h, tile_w, num_instances)
        
        # Combine instance masks into a single segmentation mask
        # Each instance gets a unique ID (1, 2, 3, ...)
        tile_h, tile_w = mask_tile.shape[:2]
        combined_mask = np.zeros((tile_h, tile_w), dtype=np.int32)
        
        tile_class_ids = []
        for i in range(mask_tile.shape[2]):
            instance_mask = mask_tile[:, :, i]
            if instance_mask.sum() > 0:  # Only include instances present in this tile
                combined_mask[instance_mask > 0] = i + 1  # Instance ID starts from 1
                tile_class_ids.append(class_ids[i])
        
        tile_class_ids = np.array(tile_class_ids, dtype=np.int32)
        
        # Pad tile if necessary
        h, w = img_tile.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            # Pad image
            padded_img = np.zeros((self.tile_size, self.tile_size, img_tile.shape[2]), dtype=img_tile.dtype)
            padded_img[:h, :w, :] = img_tile
            img_tile = padded_img
            
            # Pad mask
            padded_mask = np.zeros((self.tile_size, self.tile_size), dtype=combined_mask.dtype)
            padded_mask[:h, :w] = combined_mask
            combined_mask = padded_mask
        
        # Convert image to tensor
        img_tile = img_tile.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tile).permute(2, 0, 1).to(self.dtype)  # (C, H, W)
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(combined_mask).to(torch.long)  # (H, W)
        
        return {
            "pixel_values": img_tensor,
            "labels": mask_tensor,
            "class_ids": tile_class_ids
        }


class TiledSA1BDatasetLazy(Dataset):
    """
    Memory-efficient version of TiledSA1BDataset that loads images on-the-fly.
    
    Instead of loading all images during initialization, this version only
    pre-computes tile indices and loads images when needed. This is more
    memory-efficient for large datasets.
    
    Args:
        dataset_dir (str): Root directory of SA-1B dataset
        tile_size (int): Size of each tile (default: 256)
        dtype (torch.dtype): Output tensor dtype (default: torch.float16)
        ids (list, optional): List of sample IDs to load
        annotation_dir (str): Annotation directory name (default: 'annotations')
        image_dir (str): Image directory name (default: 'images')
        min_object (int): Minimum pixels for valid object (default: 0)
    """
    
    def __init__(
        self,
        dataset_dir,
        tile_size=256,
        dtype=torch.float16,
        ids=None,
        annotation_dir='annotations',
        image_dir='images',
        min_object=0,
    ):
        self.dataset_dir = dataset_dir
        self.tile_size = tile_size
        self.dtype = dtype
        self.min_object = min_object
        
        # Load SA1B dataset
        self.sa1b_dataset = SA1BDataset(
            dataset_dir=dataset_dir,
            ids=ids,
            annotation_dir=annotation_dir,
            image_dir=image_dir,
            min_object=min_object,
        )
        
        # Pre-compute tile indices only (not loading images)
        self.tile_index = []
        self._compute_tile_indices()
        
        logger.info(
            f"TiledSA1BDatasetLazy: {len(self.tile_index)} tiles from "
            f"{len(self.sa1b_dataset)} images in {dataset_dir}"
        )
    
    def _compute_tile_indices(self):
        """
        Pre-compute tile indices for all images without loading them.
        Uses image_info to get dimensions.
        """
        logger.info(f"Computing tile indices for {self.tile_size}x{self.tile_size} tiles...")
        
        for img_idx in range(len(self.sa1b_dataset)):
            # Get image dimensions from image_info (avoid loading actual image)
            image_info = self.sa1b_dataset.image_info[img_idx]['image']
            img_height = image_info['height']
            img_width = image_info['width']
            
            # Create tiles for this image
            for y in range(0, img_height, self.tile_size):
                for x in range(0, img_width, self.tile_size):
                    y_end = min(y + self.tile_size, img_height)
                    x_end = min(x + self.tile_size, img_width)
                    
                    # Store tile info: (image_idx, y, x, y_end, x_end)
                    self.tile_index.append((img_idx, y, x, y_end, x_end))
        
        logger.info(f"Computed {len(self.tile_index)} tile indices")
    
    def __len__(self):
        return len(self.tile_index)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary with:
            - pixel_values: image tile tensor (C, tile_size, tile_size)
            - labels: mask tile tensor (tile_size, tile_size) with instance IDs
            - class_ids: array of class IDs present in this tile
        """
        img_idx, y, x, y_end, x_end = self.tile_index[idx]
        
        # Load the full image and mask from SA1B dataset (on-the-fly)
        sample = self.sa1b_dataset[img_idx]
        img = sample["pixel_values"]  # PIL Image
        mask = sample["labels"]  # (H, W, num_instances)
        class_ids = sample["class_ids"]  # array of class IDs
        
        # Convert PIL image to numpy array
        img_array = np.array(img)  # (H, W, C)
        
        # Extract tile from image
        img_tile = img_array[y:y_end, x:x_end, :]
        
        # Extract tile from mask
        mask_tile = mask[y:y_end, x:x_end, :]  # (tile_h, tile_w, num_instances)
        
        # Combine instance masks into a single segmentation mask
        # Each instance gets a unique ID (1, 2, 3, ...)
        tile_h, tile_w = mask_tile.shape[:2]
        combined_mask = np.zeros((tile_h, tile_w), dtype=np.int32)
        
        tile_class_ids = []
        for i in range(mask_tile.shape[2]):
            instance_mask = mask_tile[:, :, i]
            if instance_mask.sum() > 0:  # Only include instances present in this tile
                combined_mask[instance_mask > 0] = i + 1  # Instance ID starts from 1
                tile_class_ids.append(class_ids[i])
        
        tile_class_ids = np.array(tile_class_ids, dtype=np.int32)
        
        # Pad tile if necessary
        h, w = img_tile.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            # Pad image
            padded_img = np.zeros((self.tile_size, self.tile_size, img_tile.shape[2]), dtype=img_tile.dtype)
            padded_img[:h, :w, :] = img_tile
            img_tile = padded_img
            
            # Pad mask
            padded_mask = np.zeros((self.tile_size, self.tile_size), dtype=combined_mask.dtype)
            padded_mask[:h, :w] = combined_mask
            combined_mask = padded_mask
        
        # Convert image to tensor
        img_tile = img_tile.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tile).permute(2, 0, 1).to(self.dtype)  # (C, H, W)
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(combined_mask).to(torch.long)  # (H, W)
        
        return {
            "pixel_values": img_tensor,
            "labels": mask_tensor,
            "class_ids": tile_class_ids
        }
