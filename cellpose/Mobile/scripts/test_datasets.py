from cellpose.Mobile.utils.dataset_utils import get_train_val_dataset_formatted

train, val = get_train_val_dataset_formatted()

print(f"train length {len(train)} val length {len(val)}")