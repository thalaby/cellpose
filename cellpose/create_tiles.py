#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_tile_index_for_image(
    sample_idx: int,
    image_path: Path,
    tile_size: int,
    ds_idx: int = 0,
    include_masks: bool = False,
    masks_filter: str = '_masks',
    img_suffix: str = '_img',
):
    """
    Compute tile index entries for a single image.

    Returns a numpy array of shape (num_tiles, 6) or (num_tiles, 7):
        Without masks: (ds_idx, sample_idx, y, x, H, W)
        With masks: (ds_idx, sample_idx, y, x, H, W, mask_sample_idx)
        
    mask_sample_idx is -1 if no mask file found, otherwise the index of the mask file.
    """
    try:
        # Use PIL to support many formats (PNG, JPG, TIFF, etc.)
        with Image.open(image_path) as img:
            # Ensure it's loaded enough to get size
            W, H = img.size  # PIL gives (W, H)
    except Exception as e:
        # On failure, return empty array and the error
        cols = 7 if include_masks else 6
        return sample_idx, image_path, None, np.zeros((0, cols), dtype=np.int64), str(e)

    # Find corresponding mask file if include_masks is True
    mask_path = None
    mask_sample_idx = -1
    
    if include_masks:
        # Get the image file stem and extension
        img_stem = image_path.stem
        img_dir = image_path.parent
        
        # Check if the image name ends with img_suffix
        if img_stem.endswith(img_suffix):
            # Extract the prefix (e.g., "000" from "000_img")
            prefix = img_stem[:-len(img_suffix)]
            mask_stem = prefix + masks_filter
            
            # Try to find mask file with any supported extension
            supported_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
            for ext in supported_extensions:
                potential_mask = img_dir / (mask_stem + ext)
                if potential_mask.is_file():
                    mask_path = potential_mask
                    mask_sample_idx = sample_idx  # Use same index for now
                    break

    entries = []

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            if include_masks:
                entries.append((ds_idx, sample_idx, y, x, H, W, mask_sample_idx))
            else:
                entries.append((ds_idx, sample_idx, y, x, H, W))

    tile_index = np.array(entries, dtype=np.int64)
    return sample_idx, image_path, mask_path, tile_index, None


def collect_image_paths(input_dir: Path, recursive: bool = False, img_suffix: str = '_img', include_masks: bool = False):
    """
    Collect all files in a directory that look like images.
    If include_masks is True, only collect files that end with img_suffix.
    """
    if recursive:
        all_paths = sorted(p for p in input_dir.rglob("*") if p.is_file())
    else:
        all_paths = sorted(p for p in input_dir.iterdir() if p.is_file())
    
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
        default="tile_index.npy",
        help="Output .npy file for the combined tile_index (default: tile_index.npy).",
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
        help="Include mask file paths in the tile index. Expects image files to end with --img_suffix.",
    )
    parser.add_argument(
        "--masks_filter",
        type=str,
        default='_masks',
        help="Suffix for mask files (default: '_masks'). E.g., if image is '000_img.png', mask is '000_masks.png'.",
    )
    parser.add_argument(
        "--img_suffix",
        type=str,
        default='_img',
        help="Suffix for image files (default: '_img'). Used when --include_masks is enabled.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    tile_size = args.B
    ds_idx = args.ds_idx
    out_path = Path(args.output)

    image_paths = collect_image_paths(input_dir, recursive=args.recursive, 
                                     img_suffix=args.img_suffix, 
                                     include_masks=args.include_masks)
    if not image_paths:
        raise SystemExit(f"No files found in directory: {input_dir}")

    print(f"Found {len(image_paths)} file(s) in {input_dir}")
    if args.include_masks:
        print(f"Looking for mask files with suffix '{args.masks_filter}' for images with suffix '{args.img_suffix}'")
    print(f"Using {args.workers or 'all available'} worker(s).")

    all_tile_indices = []
    mask_info = {}  # Map sample_idx -> mask_path
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
            ): (sample_idx, img_path)
            for sample_idx, img_path in enumerate(image_paths)
        }

        for future in tqdm(as_completed(futures), total=len(futures), unit="image", desc="Computing tile_index"):
            sample_idx, img_path, mask_path, tile_index, error = future.result()
            if error is not None:
                errors.append((img_path, error))
            else:
                all_tile_indices.append(tile_index)
                if mask_path is not None:
                    mask_info[sample_idx] = str(mask_path)

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
        num_with_masks = sum(1 for idx in mask_info if mask_info[idx] is not None)
        print(f"Found masks for {num_with_masks}/{len(image_paths)} images")
        print(f"Mask info saved to: {mask_info_path}")

    if errors:
        print("\nSome files could not be processed:")
        for img_path, err in errors[:20]:
            print(f" - {img_path}: {err}")
        if len(errors) > 20:
            print(f" ... and {len(errors) - 20} more.")


if __name__ == "__main__":
    main()
