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
):
    """
    Compute tile index entries for a single image.

    Returns a numpy array of shape (num_tiles, 6):
        (ds_idx, sample_idx, y, x, H, W)
    """
    try:
        # Use PIL to support many formats (PNG, JPG, TIFF, etc.)
        with Image.open(image_path) as img:
            # Ensure it's loaded enough to get size
            W, H = img.size  # PIL gives (W, H)
    except Exception as e:
        # On failure, return empty array and the error
        return sample_idx, image_path, np.zeros((0, 6), dtype=np.int64), str(e)

    entries = []

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            entries.append((ds_idx, sample_idx, y, x, H, W))

    tile_index = np.array(entries, dtype=np.int64)
    return sample_idx, image_path, tile_index, None


def collect_image_paths(input_dir: Path, recursive: bool = False):
    """
    Collect all files in a directory that look like images.
    We don't strictly filter by extension; we just try to open with PIL later.
    """
    if recursive:
        paths = sorted(p for p in input_dir.rglob("*") if p.is_file())
    else:
        paths = sorted(p for p in input_dir.iterdir() if p.is_file())
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
        help="Tile size B (tiles will be BxB).",
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

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    tile_size = args.B
    ds_idx = args.ds_idx
    out_path = Path(args.output)

    image_paths = collect_image_paths(input_dir, recursive=args.recursive)
    if not image_paths:
        raise SystemExit(f"No files found in directory: {input_dir}")

    print(f"Found {len(image_paths)} file(s) in {input_dir}")
    print(f"Using {args.workers or 'all available'} worker(s).")

    all_tile_indices = []
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
            ): (sample_idx, img_path)
            for sample_idx, img_path in enumerate(image_paths)
        }

        for future in tqdm(as_completed(futures), total=len(futures), unit="image", desc="Computing tile_index"):
            sample_idx, img_path, tile_index, error = future.result()
            if error is not None:
                errors.append((img_path, error))
            else:
                all_tile_indices.append(tile_index)

    if not all_tile_indices:
        raise SystemExit("No valid images were processed successfully; tile_index is empty.")

    # Concatenate all per-image tile_index arrays into one big array
    combined_tile_index = np.concatenate(all_tile_indices, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, combined_tile_index)

    print(f"\nDone. Combined tile_index shape: {combined_tile_index.shape}")
    print(f"Saved to: {out_path}")

    if errors:
        print("\nSome files could not be processed:")
        for img_path, err in errors[:20]:
            print(f" - {img_path}: {err}")
        if len(errors) > 20:
            print(f" ... and {len(errors) - 20} more.")


if __name__ == "__main__":
    main()
