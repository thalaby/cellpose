#!/usr/bin/env python3
"""
Tile SA-1B dataset with flow computation.

This script loads images and masks from SA-1B dataset using SA1BDataset,
creates tiles, computes flows from masks, and saves:
- tile_index.npy in the images directory
- flow TIFF files in the annotations directory (matching mask names)
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import numpy as np
from tqdm import tqdm
import tifffile

from cellpose.Mobile.utils.sa import SA1BDataset
from cellpose import dynamics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_flows_from_mask(mask, device=None, niter=None):
    """
    Compute flows from a mask array using cellpose dynamics.
    
    Args:
        mask: numpy array (H, W, num_instances) with binary instance masks
        device: Device to use for computation (None = auto-select)
        niter: Number of iterations for flow computation
        
    Returns:
        flows: numpy array of shape (4, H, W) containing:
               [0] = combined labels (instance IDs)
               [1] = cell probability / distance transform
               [2] = Y flow
               [3] = X flow
    """
    if mask.ndim == 3 and mask.shape[2] > 0:
        # Combine instance masks into single segmentation
        H, W, num_instances = mask.shape
        combined_mask = np.zeros((H, W), dtype=np.int32)
        
        for i in range(num_instances):
            instance_mask = mask[:, :, i]
            if instance_mask.sum() > 0:
                # Assign unique ID to each instance (1, 2, 3, ...)
                combined_mask[instance_mask > 0] = i + 1
    elif mask.ndim == 2:
        combined_mask = mask.astype(np.int32)
    else:
        # Empty mask
        return None
    
    # Check if mask has any objects
    if combined_mask.max() == 0:
        return None
    
    # Compute flows using cellpose dynamics
    # labels_to_flows expects a list of [1, H, W] arrays
    try:
        flows = dynamics.labels_to_flows(
            [combined_mask[np.newaxis, :, :]],
            device=device,
            redo_flows=True,
            niter=niter,
            return_flows=True
        )
        
        # flows[0] has shape (4, H, W)
        return flows[0]
    except Exception as e:
        logger.error(f"Error computing flows: {e}")
        return None


def process_single_sample(args):
    """
    Process a single SA-1B sample: load image/mask, compute tiles, compute flows.
    
    Args:
        args: tuple of (img_idx, sample, tile_size, annotation_dir, image_dir, 
                       compute_flows, device, niter, sample_id)
    
    Returns:
        tuple: (img_idx, tile_entries, flow_saved, error_msg)
    """
    (img_idx, sample, tile_size, annotation_dir, image_dir, 
     compute_flows, device, niter, sample_id) = args
    
    try:
        # Load image and mask
        img = sample["pixel_values"]  # PIL Image
        mask = sample["labels"]  # (H, W, num_instances)
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Compute flows if requested
        flow_saved = False
        if compute_flows and mask.shape[2] > 0:
            flows = compute_flows_from_mask(mask, device=device, niter=niter)
            
            if flows is not None:
                # Save flows as TIFF in annotations directory
                flow_filename = f"{sample_id}_flows.tif"
                flow_path = os.path.join(annotation_dir, flow_filename)
                
                # Ensure directory exists
                os.makedirs(annotation_dir, exist_ok=True)
                
                # Save as TIFF with float32 precision
                tifffile.imwrite(flow_path, flows.astype(np.float32))
                flow_saved = True
        
        # Create tile entries
        tile_entries = []
        for y in range(0, img_height, tile_size):
            for x in range(0, img_width, tile_size):
                y_end = min(y + tile_size, img_height)
                x_end = min(x + tile_size, img_width)
                
                # Store: (img_idx, y, x, y_end, x_end, img_height, img_width)
                tile_entries.append((img_idx, y, x, y_end, x_end, img_height, img_width))
        
        return (img_idx, tile_entries, flow_saved, None)
        
    except Exception as e:
        return (img_idx, [], False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Tile SA-1B dataset and compute flows from masks"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root directory of SA-1B dataset (contains 'images' and 'annotations' subdirs)",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="Tile size (tiles will be tile_size x tile_size). Default: 256",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="annotations",
        help="Annotation directory name (relative to dataset_dir). Default: 'annotations'",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="Image directory name (relative to dataset_dir). Default: 'images'",
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        default=None,
        help="List of sample IDs to process. If not provided, all samples will be processed.",
    )
    parser.add_argument(
        "--min_object",
        type=int,
        default=0,
        help="Minimum number of pixels for an object to be considered valid. Default: 0",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for tile_index.npy. Default: {dataset_dir}/{image_dir}/tile_index.npy",
    )
    parser.add_argument(
        "--compute_flows",
        action="store_true",
        default=True,
        help="Compute flows from masks (default: enabled)",
    )
    parser.add_argument(
        "--no_compute_flows",
        action="store_true",
        default=False,
        help="Disable flow computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for flow computation (e.g., 'cuda', 'cpu'). Default: auto-select",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=None,
        help="Number of iterations for flow computation. Default: None (uses cellpose default)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Default: 1 (sequential processing)",
    )
    
    args = parser.parse_args()
    
    # Handle compute_flows flag
    compute_flows = args.compute_flows and not args.no_compute_flows
    
    # Validate dataset directory
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    
    annotation_dir = dataset_dir / args.annotation_dir
    image_dir = dataset_dir / args.image_dir
    
    if not annotation_dir.is_dir():
        raise SystemExit(f"Annotation directory does not exist: {annotation_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {image_dir}")
    
    # Determine output path
    if args.output is None:
        output_path = image_dir / "tile_index.npy"
    else:
        output_path = Path(args.output)
    
    # Parse device if provided
    device = None
    if args.device:
        import torch
        device = torch.device(args.device)
    
    logger.info(f"Loading SA-1B dataset from {dataset_dir}")
    logger.info(f"  Image directory: {args.image_dir}")
    logger.info(f"  Annotation directory: {args.annotation_dir}")
    logger.info(f"  Tile size: {args.tile_size}x{args.tile_size}")
    logger.info(f"  Compute flows: {compute_flows}")
    logger.info(f"  Min object size: {args.min_object}")
    
    # Load SA-1B dataset
    sa1b_dataset = SA1BDataset(
        dataset_dir=str(dataset_dir),
        ids=args.ids,
        annotation_dir=args.annotation_dir,
        image_dir=args.image_dir,
        min_object=args.min_object,
    )
    
    logger.info(f"Loaded {len(sa1b_dataset)} samples from SA-1B dataset")
    
    # Get sample IDs
    sample_ids = [
        Path(sa1b_dataset.samples[i][0]).stem 
        for i in tqdm(range(len(sa1b_dataset)), desc="Extracting sample IDs", unit="sample")
    ]
    
    # Prepare arguments for parallel processing
    # Don't load samples yet - just prepare the argument tuples
    process_args = [
        (
            i,
            sa1b_dataset[i],
            args.tile_size,
            str(annotation_dir),
            str(image_dir),
            compute_flows,
            device,
            args.niter,
            sample_ids[i],
        )
        for i in tqdm(range(len(sa1b_dataset)), desc="Preparing arguments", unit="sample")
    ]
    
    # Process samples
    all_tile_entries = []
    flows_computed = 0
    errors = []
    
    if args.workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_sample, arg): arg[0]
                for arg in process_args
            }
            
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing samples",
                unit="sample"
            ):
                img_idx, tile_entries, flow_saved, error = future.result()
                
                if error is not None:
                    errors.append((img_idx, sample_ids[img_idx], error))
                else:
                    all_tile_entries.extend(tile_entries)
                    if flow_saved:
                        flows_computed += 1
    else:
        # Sequential processing
        for arg in tqdm(process_args, desc="Processing samples", unit="sample"):
            img_idx, tile_entries, flow_saved, error = process_single_sample(arg)
            
            if error is not None:
                errors.append((img_idx, sample_ids[img_idx], error))
            else:
                all_tile_entries.extend(tile_entries)
                if flow_saved:
                    flows_computed += 1
    
    # Convert to numpy array
    if not all_tile_entries:
        raise SystemExit("No valid tiles were created; tile_index is empty.")
    
    tile_index = np.array(all_tile_entries, dtype=np.int64)
    
    # Save tile index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, tile_index)
    
    logger.info(f"\n{'='*60}")
    logger.info("Done! Summary:")
    logger.info(f"  Total samples processed: {len(sa1b_dataset)}")
    logger.info(f"  Total tiles created: {len(tile_index)}")
    logger.info(f"  Tile index shape: {tile_index.shape}")
    logger.info(f"  Tile index saved to: {output_path}")
    
    if compute_flows:
        logger.info(f"  Flows computed for {flows_computed}/{len(sa1b_dataset)} samples")
        logger.info(f"  Flow files saved in: {annotation_dir}")
    
    if errors:
        logger.warning(f"\n{len(errors)} sample(s) had errors:")
        for img_idx, sample_id, error in errors[:10]:
            logger.warning(f"  Sample {img_idx} ({sample_id}): {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")
    
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
