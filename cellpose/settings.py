from pathlib import Path

DATA_PATH = Path("/storage/timorhalabi/Research/Data/CellDatasets")

TEST_DATASET_PATHS = [
            Path(DATA_PATH, 'Cellpose', 'test'),
            Path(DATA_PATH, 'CellposeN', 'test'),
            # Path(DATA_PATH, "Deepcell/tissuenet_v1.1_test.npz"),
            # Path(DATA_PATH, "livecell/images/livecell_test_images"),
            # Path(DATA_PATH, "Deepcell/DynamicNuclearNet-segmentation-v1_0/test.npz")
        ]

MODEL_PATH = (
    "/storage/timorhalabi/Research/Data/current/model/cellpose_iteration1_anisotropy"
)

SA1B_DATASET_PATH = (
    "/storage/timorhalabi/Research/cellpose/SA-1B/images"
)
CELL_TRAIN_DATASET_PATHS = [Path(DATA_PATH, 'Cellpose', 'train'),
                       Path(DATA_PATH, 'CellposeN', 'train'),
                       Path(DATA_PATH, "Deepcell/tissuenet_v1.1_train.npz"),
                       Path(DATA_PATH, "livecell/images/livecell_train_val_images"),
                       Path(DATA_PATH, "Deepcell/DynamicNuclearNet-segmentation-v1_0/train.npz"), 
                       Path(DATA_PATH, "NeurIPS/release-part1")]
# CELL_TRAIN_DATASET_PATHS = [
#                        Path(DATA_PATH, "Deepcell/tissuenet_v1.1_train.npz")]

SA1B_TRAIN_DATASET_PATH = [SA1B_DATASET_PATH]

TRAINING_ARGS = {"train_on_cellular": True,
                 "train": True,
                 "train_batch_size": 1,
                 "eval_batch_size": 1,
                 }