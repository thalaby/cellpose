# SA-1B Dataset Tiling Script

## Overview

`tile_sa1b_dataset.py` is a script that processes the SA-1B dataset to create tiles and compute flows from masks using Cellpose.

## Features

- Loads image-mask pairs from SA-1B dataset using `SA1BDataset`
- Creates tiles of specified size from images and masks
- Computes flows from masks using Cellpose dynamics
- Saves flow files as TIFF in the annotations directory
- Generates `tile_index.npy` file in the images directory

## Usage

### Basic Usage

```bash
python tile_sa1b_dataset.py --dataset_dir /path/to/SA-1B/dataset --tile_size 256
```

### Full Options

```bash
python tile_sa1b_dataset.py \
    --dataset_dir /path/to/SA-1B/dataset \
    --tile_size 256 \
    --annotation_dir annotations \
    --image_dir images \
    --compute_flows \
    --device cuda \
    --workers 4
```

## Arguments

- `--dataset_dir` (required): Root directory of SA-1B dataset containing 'images' and 'annotations' subdirectories
- `--tile_size` (default: 256): Size of tiles (tiles will be tile_size x tile_size)
- `--annotation_dir` (default: 'annotations'): Annotation directory name relative to dataset_dir
- `--image_dir` (default: 'images'): Image directory name relative to dataset_dir
- `--ids`: List of specific sample IDs to process (if not provided, all samples are processed)
- `--min_object` (default: 0): Minimum number of pixels for an object to be considered valid
- `--output`: Custom output path for tile_index.npy (default: {dataset_dir}/{image_dir}/tile_index.npy)
- `--compute_flows` (default: enabled): Compute flows from masks
- `--no_compute_flows`: Disable flow computation
- `--device`: Device for flow computation (e.g., 'cuda', 'cpu')
- `--niter`: Number of iterations for flow computation (uses Cellpose default if not specified)
- `--workers` (default: 1): Number of parallel workers for processing

## Output

### tile_index.npy
Located in `{dataset_dir}/{image_dir}/tile_index.npy`

NumPy array of shape `(num_tiles, 7)` where each row contains:
- `[0]`: img_idx - index of the source image
- `[1]`: y - starting y coordinate of tile
- `[2]`: x - starting x coordinate of tile
- `[3]`: y_end - ending y coordinate of tile
- `[4]`: x_end - ending x coordinate of tile
- `[5]`: img_height - height of the source image
- `[6]`: img_width - width of the source image

### Flow Files
Located in `{dataset_dir}/{annotation_dir}/{sample_id}_flows.tif`

TIFF files with shape `(4, H, W)` containing:
- `[0]`: Combined labels (instance IDs)
- `[1]`: Cell probability / distance transform
- `[2]`: Y flow
- `[3]`: X flow

## Example

Process a subset of SA-1B dataset with GPU acceleration:

```bash
python tile_sa1b_dataset.py \
    --dataset_dir /data/SA-1B/sa_000000 \
    --tile_size 512 \
    --device cuda \
    --workers 8 \
    --ids sa_100000 sa_100001 sa_100002
```

## Notes

- The script uses the `SA1BDataset` class from `cellpose.Mobile.utils.sa`
- Flow computation uses `cellpose.dynamics.labels_to_flows`
- For large datasets, consider using multiple workers for parallel processing
- Flow files are saved with float32 precision to save disk space while maintaining accuracy
