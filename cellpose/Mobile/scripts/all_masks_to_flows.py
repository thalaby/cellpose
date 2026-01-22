from cellpose import dynamics
from cellpose.Mobile.utils.settings import CELL_TRAIN_DATASET_PATHS_LABELED
import os
import numpy as np
import tifffile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def process_mask_to_flow(mask_path, output_path, device=None):
    """
    Process a single mask file and compute flows.
    
    Args:
        mask_path: Path to mask TIFF file
        output_path: Path to save flow TIFF file
        device: torch device for GPU computation
    
    Returns:
        tuple: (success: bool, mask_path: str, error_msg: str or None)
    """
    try:
        # Check if flow file already exists
        if os.path.exists(output_path):
            logger.debug(f"Flow already exists, skipping: {output_path}")
            return True, str(mask_path), None
        
        # Load mask
        mask = tifffile.imread(str(mask_path))
        
        # Ensure mask is 2D
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[-1] == 1:
                mask = mask[:, :, 0]
            else:
                mask = mask[:, :, 0] if mask.shape[-1] > 1 else mask[0]
        
        # Check if mask has any cells
        if mask.max() == 0:
            logger.warning(f"Empty mask, saving zero flows: {mask_path}")
            # Save zero flows
            flows = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)
            tifffile.imwrite(str(output_path), flows)
            return True, str(mask_path), None
        
        # Compute flows using cellpose dynamics
        # masks_to_flows returns (dY, dX, cellprob, p)
        # We want dY and dX which are the flow fields
        flows = dynamics.labels_to_flows(mask[np.newaxis, :, :], device=device, niter=None)
        
        # flows[0] has shape (1, 3, H, W) where channels are [cellprob, dY, dX]
        # Extract flow_y (channel 1) and flow_x (channel 2)
        flow_y = flows[0][1, :, :]  # (H, W)
        flow_x = flows[0][2, :, :]  # (H, W)
        
        # Stack into (2, H, W) format
        flow_output = np.stack([flow_y, flow_x], axis=0).astype(np.float32)
        
        # Save flows
        tifffile.imwrite(str(output_path), flow_output)
        
        return True, str(mask_path), None
        
    except Exception as e:
        error_msg = f"Error processing {mask_path}: {str(e)}"
        logger.error(error_msg)
        return False, str(mask_path), error_msg


def find_mask_files(dataset_paths, mask_suffix="_mask"):
    """
    Find all mask files in the given dataset paths.
    
    Args:
        dataset_paths: List of dataset directory paths
        mask_suffix: Suffix for mask files (default: "_mask")
    
    Returns:
        list: List of (mask_path, flow_output_path) tuples
    """
    mask_flow_pairs = []
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            continue
        
        # Find all mask TIFF files
        mask_files = list(dataset_path.glob(f"*{mask_suffix}.tif*"))
        
        logger.info(f"Found {len(mask_files)} mask files in {dataset_path}")
        
        for mask_path in mask_files:
            # Generate output path by replacing mask_suffix with _flow
            stem = mask_path.stem.replace(mask_suffix, "")
            flow_filename = f"{stem}_flow.tif"
            flow_path = mask_path.parent / flow_filename
            
            mask_flow_pairs.append((mask_path, flow_path))
    
    return mask_flow_pairs


def main():
    import torch
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Precompute flows for all mask files')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=os.cpu_count() or 4,
        help=f'Number of worker threads (default: {os.cpu_count() or 4})'
    )
    args = parser.parse_args()
    
    # Check for GPU
    device = torch.device('cuda') if torch.cuda.is_available() else None
    if device:
        logger.info(f"Using GPU for flow computation")
    else:
        logger.info("Using CPU for flow computation (this will be slower)")
    
    # Find all mask files
    logger.info("Scanning for mask files...")
    mask_flow_pairs = find_mask_files(CELL_TRAIN_DATASET_PATHS_LABELED)
    
    total_files = len(mask_flow_pairs)
    logger.info(f"Found {total_files} total mask files to process")
    
    if total_files == 0:
        logger.warning("No mask files found!")
        return
    
    # Process files with multithreading
    num_threads = args.num_workers
    logger.info(f"Using {num_threads} threads for parallel processing")
    
    success_count = 0
    error_count = 0
    errors = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_mask_to_flow, mask_path, flow_path, device): (mask_path, flow_path)
            for mask_path, flow_path in mask_flow_pairs
        }
        
        # Process results with progress bar
        with tqdm(total=total_files, desc="Computing flows") as pbar:
            for future in as_completed(futures):
                success, mask_path, error_msg = future.result()
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append((mask_path, error_msg))
                
                pbar.update(1)
                pbar.set_postfix({'Success': success_count, 'Errors': error_count})
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Flow computation complete!")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")
    
    if errors:
        logger.error("\nErrors encountered:")
        for mask_path, error_msg in errors[:10]:  # Show first 10 errors
            logger.error(f"  {mask_path}: {error_msg}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
    
    logger.info("="*60)


if __name__ == "__main__":
    main()
