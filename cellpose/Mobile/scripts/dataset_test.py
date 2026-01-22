from cellpose.Mobile.utils.dataset_utils import get_train_val_dataset_formatted
import time
from tqdm import tqdm
from random import randint, random
import tifffile as tiff

train, val = get_train_val_dataset_formatted(flows=False, test_mode=True, minimal=False)
iterations = 1
problematic_indices = [10, 11, 20, 25, 26]
for ds in train:
    if train.index(ds) not in problematic_indices:
        print(f"Skipping dataset {train.index(ds)} it is good.")
        continue
    print(f"Train dataset: {len(ds)} samples")
    total_time = 0
    max_time = 0
    for i in tqdm(range(iterations)):
        t0 = time.time()
        idx = randint(0, len(ds)-1)
        sample = ds[idx]
        t1 = time.time()
        elapsed = t1 - t0
        total_time += elapsed
        if elapsed > max_time:
            max_time = elapsed
    tiff.imwrite(f"test_images/test_output_train_dataset_img_{train.index(ds)}.tif", sample['pixel_values'].numpy())
    tiff.imwrite(f"test_images/test_output_train_dataset_mask_{train.index(ds)}.tif", sample['labels'].numpy())
    print(f"Train datasets average time per sample: {total_time / (iterations)}")
    print(f"Train datasets max time per sample: {max_time}")

