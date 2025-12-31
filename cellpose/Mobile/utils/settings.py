from pathlib import Path

DATA_PATH = Path("/storage/timorhalabi/Research/Data/CellDatasets/Formatted")


MODEL_PATH = (
    "/storage/timorhalabi/Research/Data/current/model/cellpose_iteration1_anisotropy"
)

SA1B_DATASET_PATH = (
    "/storage/timorhalabi/Research/cellpose/SA-1B"
)
TEST_DATASET_PATHS = [
            ("Cellpose", Path(DATA_PATH, 'Cellpose', 'test')),
            ("Cellpose Nucleus", Path(DATA_PATH, 'CellposeN', 'test')),
            ("CPM17", Path(DATA_PATH, 'CPM15', 'cpm17', 'test', 'Images')),
            ("cpm-kumars", Path(DATA_PATH, 'CPM15', 'kumar', 'test_same', 'Images')),
            ("cpm-kumard", Path(DATA_PATH, 'CPM15', 'kumar', 'test_diff', 'Images')),
            ("DeepBacs", Path(DATA_PATH, 'DeepBacs', 'test', 'images')),
            ("kaggle", Path(DATA_PATH, 'kaggle_bccd', 'test', 'images')),
            ("livecell", Path(DATA_PATH, 'livecell', 'test')),
            ("MoNuSAC", Path(DATA_PATH, 'MoNuSAC', 'test', 'images')),
            ("MoNuSeg", Path(DATA_PATH, 'MoNuSeg', 'Test')),
            ("Deepcell-Tissuenet", Path(DATA_PATH, "Deepcell/tissuenet", 'test')),
            ("Deepcell-DynamicNuclearNet", Path(DATA_PATH, "Deepcell/DynamicNuclearNet-segmentation-v1_0", 'test'))
        ]
CELL_TRAIN_DATASET_PATHS_LABELED = [
                       Path(DATA_PATH, 'Cellpose', 'train'),
                       Path(DATA_PATH, 'CellposeN', 'train'),
                       Path(DATA_PATH, 'CPM15', 'cpm15', 'Images'),
                       Path(DATA_PATH, 'CPM15', 'cpm17', 'train', 'Images'),
                       Path(DATA_PATH, 'CPM15', 'kumar', 'train', 'Images'),
                       Path(DATA_PATH, 'CPM15', 'tnbc', 'Images'),
                       Path(DATA_PATH, 'CryoNuSeg', 'images'),
                       Path(DATA_PATH, 'DeepBacs', 'training', 'images'),
                       Path(DATA_PATH, "Deepcell/DynamicNuclearNet-segmentation-v1_0", 'train'), 
                       Path(DATA_PATH, "Deepcell/tissuenet", 'train'), 
                       Path(DATA_PATH, "IHC-TMA", 'images'), 
                       Path(DATA_PATH, "kaggle_bccd", 'train', 'images'), 
                       Path(DATA_PATH, "kaggle_conic", 'images'), 
                       Path(DATA_PATH, "livecell", 'train'), 
                       Path(DATA_PATH, "LynSec", 'lynsec 1'),
                       Path(DATA_PATH, "LynSec", 'lynsec 2'),
                       Path(DATA_PATH, "LynSec", 'lynsec 3'),
                       Path(DATA_PATH, "MoNuSAC", 'train', "images"),
                       Path(DATA_PATH, "MoNuSeg", 'Train', 'images'),
                       Path(DATA_PATH, "nuinsseg", 'images'),
                       Path(DATA_PATH, "PanNuke", 'Fold 1', 'images', 'fold1'),
                       Path(DATA_PATH, "PanNuke", 'Fold 2', 'images', 'fold2'),
                       Path(DATA_PATH, "PanNuke", 'Fold 3', 'images', 'fold3'),
                       Path(DATA_PATH, "YeaZ", 'gold-standard-BF-V-1'),
                    #    Path(DATA_PATH, "YeaZ", 'gold-standard-PhC-plus-2'), 3D images
                       Path(DATA_PATH, "NeurIPS", 'Training-labeled', 'images'),
                       ]

CELL_TRAIN_DATASET_PATHS_UNLABELED = [
                       Path(DATA_PATH, "NeurIPS", 'release-part1'),
                       Path(DATA_PATH, "NeurIPS", 'train-unlabeled-part2'),]

CELL_TRAIN_DATASET_PATHS_DECODER = [Path(DATA_PATH, "CellposeNDecoder/train"),
                                    Path(DATA_PATH, "CellposeDecoder/train")]
# CELL_TRAIN_DATASET_PATHS = [
                    #    "/storage/timorhalabi/Research/cellpose/SA-mini/images"]

SA1B_TRAIN_DATASET_PATH = [SA1B_DATASET_PATH + "/images"]

TRAINING_ARGS = {"train_on_cellular": True,
                 "test_original_cellpose": True,
                 "test_trained_model": True,
                 "train": False,
                 "train_batch_size": 1,
                 "eval_batch_size": 1,
                 "eval_log_steps": 500
                 }