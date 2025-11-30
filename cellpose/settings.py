from pathlib import Path

DATA_PATH = Path("/storage/timorhalabi/Research/Data/CellDatasets")

TEST_DATASET_PATHS = [
            ("Cellpose", Path(DATA_PATH, 'Cellpose', 'test')),
            ("Cellpose Nucleus", Path(DATA_PATH, 'CellposeN', 'test')),
            ("MoNuSeg", Path(DATA_PATH, 'MoNuSeg', 'Test')),
            ("MoNuSAC", Path(DATA_PATH, 'MoNuSAC', 'test')),
            ("kaggle_bccd", Path(DATA_PATH, 'kaggle_bccd', 'test')),
            ("Tissuenet", Path(DATA_PATH, "Deepcell/tissuenet_v1.1_test.npz")),
            # ("livecell", Path(DATA_PATH, "livecell/images/livecell_test_images")),
            ("DynamicNuclearNet", Path(DATA_PATH, "Deepcell/DynamicNuclearNet-segmentation-v1_0/test.npz"))
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
                       Path(DATA_PATH, "MoNuSeg/Train/Tissue Images"), 
                       Path(DATA_PATH, "MoNuSAC/images"), 
                       Path(DATA_PATH, "kaggle_bccd/train"), 
                       Path(DATA_PATH, "Deepcell/DynamicNuclearNet-segmentation-v1_0/train.npz"), 
                       Path(DATA_PATH, "NeurIPS/release-part1")]
CELL_EVAL_DATASET_PATHS = [Path(DATA_PATH, "Deepcell/tissuenet_v1.1_val.npz"),
                       Path(DATA_PATH, "Deepcell/DynamicNuclearNet-segmentation-v1_0/val.npz")]
# CELL_TRAIN_DATASET_PATHS = [
                    #    "/storage/timorhalabi/Research/cellpose/SA-mini/images"]

SA1B_TRAIN_DATASET_PATH = [SA1B_DATASET_PATH]

TRAINING_ARGS = {"train_on_cellular": True,
                 "train": True,
                 "train_batch_size": 1,
                 "eval_batch_size": 1,
                 "eval_log_steps": 500
                 }