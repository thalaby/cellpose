from vit_sam import Transformer
from vit_tiny import SAMStyleTinyViTEncoder, SAMStyleTinyViTDecoder
from sa import SA1BDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from train_utils import (
    ImageMaskDataset,
    DistillationDatasetWrapper,
    DistillationModel,
    DistillationTrainer,
    TransformerWrapper,
    StudentSegmentationModel,
    CellposeCustomModel,
    TiffImageDataset,
    DistillationDatasetWrapperIndex,
    TiledImageDirDataset
)
from torch import nn
from transformers import TrainingArguments
from logging import getLogger
from cellpose import metrics
from pathlib import Path
import torch
import logging
import numpy as np
import wandb

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = getLogger(__name__)
MODEL_PATH = (
    "/storage/timorhalabi/Research/Data/current/model/cellpose_iteration1_anisotropy"
)

CELLPOSE_DATASET_PATH = (
    "/storage/timorhalabi/Research/Data/CellDatasets/Cellpose"
)

CELLPOSEN_DATASET_PATH = (
    "/storage/timorhalabi/Research/Data/CellDatasets/CellposeN"
)

SA1B_DATASET_PATH = (
    "/storage/timorhalabi/Research/cellpose/SA-1B/tiled_images"
)


def dice_coef(pred, target, eps=1e-6):
    # pred, target: (B, 1, H, W), 0/1
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def evaluate_on_masks(
    student_encoder, student_decoder, val_dataset, device="cuda", batch_size=1, teacher_model=None
):
    tiny_network = StudentSegmentationModel(student_encoder, student_decoder).to(device)
    model = CellposeCustomModel(gpu=True, nchan=3, use_bfloat16=False, custom_net=tiny_network)
    

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    masks_gt_all =[]
    masks_pred_all =[]
    with torch.no_grad():
        for batch in loader:
            x = batch[0][0].permute((1, 2, 0)).to(device)  # (B, 3, 256, 256)
            masks_gt = batch[1][0]  # (B, 1, 256, 256), 0/1
            masks_gt = masks_gt.squeeze(1).numpy().astype(np.int16)
            masks, flows, styles = model.eval(x)  # (B, 3, 256, 256) assumed
            ap, tp, fp, fn = metrics.average_precision(masks_gt, masks)
            print(ap[0], tp[0], fp[0], fn[0])
            masks_gt_all.append(masks_gt)
            masks_pred_all.append(masks)

    threshold = np.arange(0.5, 1, 0.05)
    
    ap, tp, fp, fn = metrics.average_precision(masks_gt_all, masks_pred_all, threshold=threshold)
    return {"ap": ap, "tp": tp, "fp": fp, "fn": fn}


def load_models(device="cuda", dtype=torch.float16):
    """Load student and teacher encoders"""
    student_encoder = SAMStyleTinyViTEncoder(device=device, dtype=dtype)
    student_decoder = SAMStyleTinyViTDecoder(device=device, dtype=dtype)
    teacher_model = Transformer(
        backbone="vit_l", ps=8, nout=3, bsize=256, rdrop=0.4, dtype=dtype
    )
    teacher_model.load_model(PATH=MODEL_PATH, device=device)
    teacher_encoder = TransformerWrapper(teacher_model.encoder)
    student_decoder.out.weight.data = teacher_model.out.weight.data.clone()
    student_decoder.out.bias.data = teacher_model.out.bias.data.clone()
    student_decoder = student_decoder.to(device, dtype=dtype)
    return student_encoder, teacher_encoder, student_decoder, teacher_model


def create_training_args(output_dir="./distillation_output", **kwargs):
    """Create training arguments for the Trainer"""
    default_args = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 1,
        "learning_rate": 1e-4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 100,
        "save_steps": 5000,
        "save_total_limit": 3,
        "seed": 42,
        "fp16": False,  # Disable FP16 since models are already float16
        "gradient_accumulation_steps": 1,
        "remove_unused_columns": False,
        "report_to": "wandb",
        "logging_steps": 10,
        "dataloader_num_workers": 8,
        "ddp_find_unused_parameters": False,
    }
    default_args.update(kwargs)
    return TrainingArguments(**default_args)


def main(
    dataset_dir="../SA-1B/images",
    output_dir="./distillation_output",
    num_epochs=3,
    batch_size=64,
    learning_rate=1e-4,
    batch_size_eval=1,
):
    """Main training function"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32


    logger.info(f"Using device: {device}, dtype: {dtype}")
    train = True
    # Load encoders
    logger.info("Loading student and teacher encoders...")
    student_encoder, teacher_encoder, student_decoder, teacher_model = load_models(
        device=device, dtype=dtype
    )

    # Create distillation model
    logger.info("Creating distillation model...")
    distillation_model = DistillationModel(student_encoder, teacher_encoder)
    distillation_model.to(device, dtype=dtype)

    # Load dataset
    logger.info(f"Loading datasets ...")

    logger.info(f"Loading datasets from {dataset_dir}...")
    # sa1b_dataset = TiffImageDataset(SA1B_DATASET_PATH)
    sa1b_dataset = TiledImageDirDataset(dataset_dir, dtype=dtype)
    cellpose_train_dataset = TiledImageDirDataset(root_dir=Path(CELLPOSE_DATASET_PATH, 'train'), dtype=dtype)
    cellposen_train_dataset = TiledImageDirDataset(root_dir=Path(CELLPOSEN_DATASET_PATH, 'train'), dtype=dtype)
    logger.info(f"sa1b loaded with {len(sa1b_dataset)}, cellpose train dataset loaded with {len(cellpose_train_dataset)}, cellposen train dataset loaded with {len(cellposen_train_dataset)} samples.")

    logger.info(f"cellpose train dataset loaded with {len(cellpose_train_dataset)} samples.")
    if train:
        # train_dataset = DistillationDatasetWrapper(processed_datasets=[sa1b_dataset], unprocessed_datasets=[cellpose_train_dataset, cellposen_train_dataset], dtype=dtype)
        train_dataset = DistillationDatasetWrapperIndex(datasets=[sa1b_dataset, cellpose_train_dataset, cellposen_train_dataset])

        logger.info(f"Train tiles: {len(train_dataset)}")


        # Create training arguments
        training_args = create_training_args(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Create trainer
        logger.info("Creating trainer...")
        trainer = DistillationTrainer(
            model=distillation_model,
            args=training_args,
            train_dataset=train_dataset,
            loss_fn=nn.MSELoss(),
        )

        # Start training
        logger.info("Starting distillation training...")
        trainer.train()
        student_encoder = distillation_model.student_encoder
    else:
        student_encoder.load_state_dict(torch.load('distillation_output/student_encoder.pt'))

    # Save the student encoder
    logger.info(f"Saving student encoder to {output_dir}/student_encoder.pt")
    torch.save(student_encoder.state_dict(), f"{output_dir}/student_encoder.pt")

    logger.info("Starting Evaluation..")
    logger.info("Creating validation segmentation dataset...")
    val_seg_dataset = ImageMaskDataset(root_dir=Path(CELLPOSE_DATASET_PATH, 'test'), dtype=dtype)
    metrics = evaluate_on_masks(
        student_encoder, student_decoder, val_seg_dataset, device=device, batch_size=batch_size_eval, teacher_model=teacher_model)
    
    np.savez(f"{output_dir}/segmentation_metrics.npz", **metrics)
    mean_ap = metrics["ap"].mean()
    logger.info(f"Segmentation eval: {mean_ap}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
