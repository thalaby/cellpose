from vit_sam import Transformer
from vit_tiny import SAMStyleTinyViTEncoder, SAMStyleTinyViTDecoder
from sa import SA1BDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from train_utils import (
    SegmentationTileDataset,
    DistillationDatasetWrapper,
    DistillationModel,
    DistillationTrainer,
    TransformerWrapper,
    StudentSegmentationModel,
)
import torch
from torch import nn
from transformers import TrainingArguments
from logging import getLogger
import logging

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = getLogger(__name__)
MODEL_PATH = (
    "/storage/timorhalabi/Research/Data/current/model/cellpose_iteration1_anisotropy"
)


def dice_coef(pred, target, eps=1e-6):
    # pred, target: (B, 1, H, W), 0/1
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def evaluate_on_masks(
    student_encoder, student_decoder, val_dataset, device="cuda", batch_size=4
):
    model = StudentSegmentationModel(student_encoder, student_decoder).to(device)
    model.eval()

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)  # (B, 3, 256, 256)
            y_true = batch["labels"].to(device)  # (B, 1, 256, 256), 0/1

            logits = model(x)  # (B, 1, 256, 256) assumed
            probs = torch.sigmoid(logits)
            y_pred = (probs > 0.5).float()

            # Dice
            import ipdb; ipdb.set_trace()
            dice = dice_coef(y_pred, y_true)

            # IoU
            intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
            union = ((y_pred + y_true) > 0).float().sum(dim=(1, 2, 3))
            iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()

            total_dice += dice
            total_iou += iou
            total_batches += 1

    avg_dice = total_dice / max(total_batches, 1)
    avg_iou = total_iou / max(total_batches, 1)
    print(f"Eval on val set: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}")
    return {"dice": avg_dice, "iou": avg_iou}


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
    return student_encoder, teacher_encoder, student_decoder


def create_training_args(output_dir="./distillation_output", **kwargs):
    """Create training arguments for the Trainer"""
    default_args = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "learning_rate": 1e-4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 100,
        "save_steps": 500,
        "save_total_limit": 3,
        "seed": 42,
        "fp16": False,  # Disable FP16 since models are already float16
        "gradient_accumulation_steps": 1,
        "remove_unused_columns": False,
    }
    default_args.update(kwargs)
    return TrainingArguments(**default_args)


def main(
    dataset_dir="../SA-mini",
    output_dir="./distillation_output",
    num_epochs=3,
    batch_size=1,
    learning_rate=1e-4,
    batch_size_eval=4,
):
    """Main training function"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Load encoders
    logger.info("Loading student and teacher encoders...")
    student_encoder, teacher_encoder, student_decoder = load_models(
        device=device, dtype=dtype
    )

    # Create distillation model
    logger.info("Creating distillation model...")
    distillation_model = DistillationModel(student_encoder, teacher_encoder)
    distillation_model.to(device, dtype=dtype)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir}...")
    sa1b_dataset = SA1BDataset(dataset_dir=dataset_dir)
    logger.info(f"Dataset loaded with {len(sa1b_dataset)} samples.")

    logger.info(f"Splitting Dataset into train and val sets...")

    n = len(sa1b_dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_base, val_base = random_split(sa1b_dataset, [n_train, n_val])
    logger.info(f"Train samples: {len(train_base)}, Val samples: {len(val_base)}")

    train_dataset = DistillationDatasetWrapper(train_base, dtype=dtype)
    val_seg_dataset = SegmentationTileDataset(val_base, dtype=dtype)

    logger.info(f"Train tiles: {len(train_dataset)}, Val tiles: {len(val_seg_dataset)}")


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
    # trainer.train()

    # Save the student encoder
    logger.info(f"Saving student encoder to {output_dir}/student_encoder.pt")
    torch.save(student_encoder.state_dict(), f"{output_dir}/student_encoder.pt")

    logger.info("Starting Evaluation..")
    metrics = evaluate_on_masks(
        student_encoder, student_decoder, val_seg_dataset, device=device, batch_size=batch_size_eval
    )
    logger.info(f"Segmentation eval: {metrics}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
