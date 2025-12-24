#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm
import tifffile

# Import cellpose dynamics for flow computation
try:
    from cellpose import dynamics
except ImportError:
    print("Warning: cellpose not found. Flow computation will be disabled.")
    dynamics = None


def compute_flows_from_mask(mask_path: Path, output_path: Path, device=None):
    """
    Compute flows from a mask file and save as TIFF.
    
    Args:
        mask_path: Path to the mask file
        output_path: Path to save the flow file
        device: Device to use for computation (None = auto-select)
        
    Returns:
        True if successful, False otherwise
    """
    if dynamics is None:
        return False
        
    try:
        # Load mask
        with Image.open(mask_path) as mask_img:
            mask = np.array(mask_img)
        
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.mean(axis=2)
        
        # Compute flows using cellpose dynamics
        # labels_to_flows expects a list of masks
        flows = dynamics.labels_to_flows([mask[np.newaxis, :, :]], device=device, return_flows=True)
        
        # flows[0] has shape (4, H, W): [labels, cellprob, dY, dX]
        flow = flows[0]
        
        # Save as TIFF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(output_path), flow.astype(np.float32))
        
        return True
    except Exception as e:
        print(f"Error computing flows for {mask_path}: {e}")
        return False


def compute_tile_index_for_image(
    sample_idx: int,
    image_path: Path,
    tile_size: int,
    ds_idx: int = 0,
    include_masks: bool = False,
    masks_filter: str = '_mask',
    img_suffix: str = '_im',
    compute_flows: bool = False,
    device=None,
):
    """
    Compute tile index entries for a single image.

    Returns a numpy array of shape (num_tiles, 6) or (num_tiles, 7):
        Without masks: (ds_idx, sample_idx, y, x, H, W)
        With masks: (ds_idx, sample_idx, y, x, H, W, mask_sample_idx)
        
    mask_sample_idx is -1 if no mask file found, otherwise the index of the mask file.
    
    If compute_flows is True, also computes and saves flow files.
    """
    try:
        # Try tifffile first for TIFF files (better support for scientific TIFFs)
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            try:
                img = tifffile.imread(str(image_path))
                if img.ndim == 2:
                    H, W = img.shape
                elif img.ndim == 3:
                    H, W = img.shape[:2]
                else:
                    raise ValueError(f"Unexpected image dimensions: {img.ndim}")
            except Exception as tiff_err:
                # Fallback to PIL
                with Image.open(image_path) as img:
                    W, H = img.size  # PIL gives (W, H)
        else:
            # Use PIL for non-TIFF formats (PNG, JPG, etc.)
            with Image.open(image_path) as img:
                W, H = img.size  # PIL gives (W, H)
    except Exception as e:
        # On failure, return empty array and the error
        cols = 7 if include_masks else 6
        return sample_idx, image_path, None, None, np.zeros((0, cols), dtype=np.int64), str(e)

    # Find corresponding mask file if include_masks is True
    mask_path = None
    flow_path = None
    mask_sample_idx = -1
    
    if include_masks:
        # Get the image file stem and extension
        img_stem = image_path.stem
        img_dir = image_path.parent
        
        # Check if the image name ends with img_suffix
        if img_stem.endswith(img_suffix):
            # Extract the prefix (e.g., "A172_Phase_C7_1_00d00h00m_1" from "A172_Phase_C7_1_00d00h00m_1_im")
            prefix = img_stem[:-len(img_suffix)]
            mask_stem = prefix + masks_filter
            
            # Look for mask file with .tif/.tiff extension (prioritizing .tif)
            potential_mask = img_dir / (mask_stem + '.tif')
            if potential_mask.is_file():
                mask_path = potential_mask
                mask_sample_idx = sample_idx  # Use same index for now
            else:
                # Try .tiff as fallback
                potential_mask = img_dir / (mask_stem + '.tiff')
                if potential_mask.is_file():
                    mask_path = potential_mask
                    mask_sample_idx = sample_idx
            
            # Compute flows if mask was found and compute_flows is enabled
            if mask_path is not None and compute_flows:
                flow_stem = prefix + '_flows'
                flow_path = img_dir / (flow_stem + '.tif')
                
                # Only compute if flow file doesn't exist
                if not flow_path.exists():
                    success = compute_flows_from_mask(mask_path, flow_path, device=device)
                    if not success:
                        flow_path = None

    entries = []

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            if include_masks:
                entries.append((ds_idx, sample_idx, y, x, H, W, mask_sample_idx))
            else:
                entries.append((ds_idx, sample_idx, y, x, H, W))

    tile_index = np.array(entries, dtype=np.int64)
    return sample_idx, image_path, mask_path, flow_path, tile_index, None


def collect_image_paths(input_dir: Path, recursive: bool = False, img_suffix: str = '_im', include_masks: bool = False):
    """
    Collect all files in a directory that look like images.
    If include_masks is True, only collect files that end with img_suffix.
    Excludes flow files (_flows.tif) and mask files (_mask.*).
    """
    if recursive:
        all_paths = sorted(p for p in input_dir.rglob("*") if p.is_file())
    else:
        all_paths = sorted(p for p in input_dir.iterdir() if p.is_file())
    
    # Filter out flow files and mask files (and .npy files)
    all_paths = [
        p for p in all_paths 
        if not p.stem.endswith('_flows') 
        and not p.stem.endswith('_mask')
        and p.suffix.lower() != '.npy'
    ]
    
    # If include_masks is True, filter only image files (ending with img_suffix)
    if include_masks:
        paths = [p for p in all_paths if p.stem.endswith(img_suffix)]
    else:
        paths = all_paths
    
    return paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute tile_index array for all images in a directory, "
            "using multiple workers."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing images of any format.",
    )
    parser.add_argument(
        "--B",
        type=int,
        required=True,
        help="Tile size B (tiles will be BxB). Required argument.",
    )
    parser.add_argument(
        "--ds-idx",
        type=int,
        default=0,
        help="Dataset index to store in tile_index entries (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npy file for the combined tile_index.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images in subdirectories.",
    )
    parser.add_argument(
        "--include_masks",
        action="store_true",
        default=False,
        help="Include mask file paths in the tile index. Expects image files to end with --img_suffix.",
    )
    parser.add_argument(
        "--masks_filter",
        type=str,
        default='_mask',
        help="Suffix for mask files (default: '_mask'). E.g., if image is 'A172_Phase_C7_1_00d00h00m_1_im.tif', mask is 'A172_Phase_C7_1_00d00h00m_1_mask.tif'.",
    )
    parser.add_argument(
        "--img_suffix",
        type=str,
        default='_im',
        help="Suffix for image files (default: '_im'). Used when --include_masks is enabled.",
    )
    parser.add_argument(
        "--compute_flows",
        action="store_true",
        default=False,
        help="Compute flow files from masks (requires cellpose). Flows will be saved as {prefix}_flows.tif",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for flow computation (e.g., 'cuda', 'cpu'). Default: auto-select",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    tile_size = args.B
    ds_idx = args.ds_idx
    if args.output is None:
        out_path = input_dir / "tile_index.npy"
    else:
        out_path = Path(args.output)
    
    # Parse device if provided
    device = None
    if args.device:
        import torch
        device = torch.device(args.device)

    image_paths = collect_image_paths(input_dir, recursive=args.recursive, 
                                     img_suffix=args.img_suffix, 
                                     include_masks=args.include_masks)
    if not image_paths:
        raise SystemExit(f"No files found in directory: {input_dir}")

    print(f"Found {len(image_paths)} file(s) in {input_dir}")
    if args.include_masks:
        print(f"Looking for mask files with suffix '{args.masks_filter}' for images with suffix '{args.img_suffix}'")
        if args.compute_flows:
            print(f"Flow computation enabled. Flows will be saved as {{prefix}}_flows.tif")
    print(f"Using {args.workers or 'all available'} worker(s).")

    all_tile_indices = []
    mask_info = {}  # Map sample_idx -> {'mask_path': str, 'flow_path': str}
    errors = []

    # Multiprocessing over images
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                compute_tile_index_for_image,
                sample_idx,
                img_path,
                tile_size,
                ds_idx,
                args.include_masks,
                args.masks_filter,
                args.img_suffix,
                args.compute_flows,
                device,
            ): (sample_idx, img_path)
            for sample_idx, img_path in enumerate(image_paths)
        }

        for future in tqdm(as_completed(futures), total=len(futures), unit="image", desc="Computing tile_index"):
            sample_idx, img_path, mask_path, flow_path, tile_index, error = future.result()
            if error is not None:
                errors.append((img_path, error))
            else:
                all_tile_indices.append(tile_index)
                if mask_path is not None:
                    mask_info[sample_idx] = {
                        'mask_path': str(mask_path),
                        'flow_path': str(flow_path) if flow_path else None
                    }

    if not all_tile_indices:
        raise SystemExit("No valid images were processed successfully; tile_index is empty.")

    # Concatenate all per-image tile_index arrays into one big array
    combined_tile_index = np.concatenate(all_tile_indices, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, combined_tile_index)

    print(f"\nDone. Combined tile_index shape: {combined_tile_index.shape}")
    print(f"Saved to: {out_path}")
    
    # Save mask information if include_masks is enabled
    if args.include_masks and mask_info:
        mask_info_path = out_path.parent / (out_path.stem + "_mask_info.npy")
        np.save(mask_info_path, mask_info)
        num_with_masks = sum(1 for idx in mask_info if mask_info[idx]['mask_path'] is not None)
        num_with_flows = sum(1 for idx in mask_info if mask_info[idx].get('flow_path') is not None)
        print(f"Found masks for {num_with_masks}/{len(image_paths)} images")
        if args.compute_flows:
            print(f"Computed flows for {num_with_flows}/{num_with_masks} masks")
        print(f"Mask info saved to: {mask_info_path}")

    if errors:
        print("\nSome files could not be processed:")
        for img_path, err in errors[:20]:
            print(f" - {img_path}: {err}")
        if len(errors) > 20:
            print(f" ... and {len(errors) - 20} more.")


if __name__ == "__main__":
    main()
