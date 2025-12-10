"""
Decoder training script for Cellpose model.

Trains the decoder component using frozen encoder features and Cellpose's segmentation loss.
"""

import logging
from cellpose.Mobile.models.train_utils import DistillationModel
import torch
import wandb

from tqdm import tqdm
from pathlib import Path
from cellpose.Mobile.models.decoder_utils import (
    StudentSegmentationModelDecoderTrain,
    DecoderTrainer,
    load_encoder,
)
from cellpose.Mobile.utils.dataset_utils import (
    get_train_val_dataset_decoder,
    get_test_dataset,
    get_train_dataset_sa1b,
)
from cellpose.Mobile.models.vit_tiny import SAMStyleTinyViTDecoder
from cellpose.Mobile.utils.utils import load_models, create_cellpose_model
from safetensors.torch import load_file

if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
    torch.multiprocessing.set_start_method("spawn")
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Custom collate function to handle variable-length class_ids."""
    # Stack fixed-size tensors
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    flows = torch.stack([item["flows"] for item in batch])

    # class_ids is variable length, so we don't stack it (not needed for training)
    # If needed in the future, could pad or use a list

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "flows": flows,
    }


def create_training_args(output_dir="./decoder_output", **kwargs):
    """Create training arguments for the Trainer"""
    from transformers import TrainingArguments

    default_args = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": 4,
        "per_device_train_batch_size": 256,
        "per_device_eval_batch_size": 256,
        "dataloader_num_workers": 16,
        "learning_rate": 1e-4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 100,
        "eval_steps": 500,
        "save_steps": 500,
        "save_total_limit": 3,
        "eval_strategy": "epoch",
        "seed": 42,
        "fp16": False,
        "gradient_accumulation_steps": 1,
        "remove_unused_columns": False,
        "report_to": "wandb",
    }
    default_args.update(kwargs)
    return TrainingArguments(**default_args)


def main(
    encoder_path="./distillation_output/student_encoder.pt",
    output_dir="./decoder_output",
    learning_rate=1e-4,
):
    """Main decoder training function"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    wandb.init(project="cellpose_distillation", name="distillation_run")

    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Load pretrained encoder
    logger.info("Loading pretrained encoder...")
    student_encoder, teacher_encoder, student_decoder, teacher_model = load_models(
        device=device, dtype=dtype
    )
    # Create distillation model
    logger.info("Creating distillation model...")
    logger.info("Loading student encoder state dict from previous SA1B training...")
    # student_encoder.load_state_dict(torch.load('distillation_output/student_encoder.pt'))
    distillation_model = DistillationModel(student_encoder, teacher_encoder)
    distillation_model.to(device, dtype=dtype)
    state_dict = load_file(
        "good_checkpoints/checkpoint-post-cell/model.safetensors", device="cpu"
    )
    distillation_model.load_state_dict(state_dict)
    distillation_model.to(device, dtype=dtype)
    # Create decoder
    logger.info("Creating decoder...")
    decoder = SAMStyleTinyViTDecoder(ps=8, nout=3)
    decoder = decoder.to(device).to(dtype)

    # Create student segmentation model
    logger.info("Creating student segmentation model...")
    student_model = StudentSegmentationModelDecoderTrain(
        distillation_model.student_encoder, student_decoder, device=device, dtype=dtype
    )
    student_model.to(device)

    # Load datasets
    logger.info("Loading training and validation datasets...")
    # train_dataset, val_dataset = get_train_val_dataset_decoder()
    train_dataset, val_dataset = get_train_dataset_sa1b()

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Create training arguments
    training_args = create_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = DecoderTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start training
    logger.info("Starting decoder training...")
    trainer.teacher_model = teacher_model.to(device, dtype)
    trainer.train()
    cellpose_model = create_cellpose_model(
        student_model.encoder, student_model.decoder, device=device
    )
    trainer.cellpose_model = cellpose_model
    # Save the decoder
    logger.info(f"Saving decoder to {output_dir}/decoder.pt")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), f"{output_dir}/decoder.pt")

    # Save the complete model
    logger.info(f"Saving complete model to {output_dir}/student_model.pt")
    torch.save(student_model.state_dict(), f"{output_dir}/student_model.pt")

    logger.info("Training completed!")

    # Optional: Run test evaluation
    logger.info("Running test evaluation...")
    test_datasets = get_test_dataset()
    trainer.test(test_dataset=test_datasets)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Cellpose decoder")
    parser.add_argument(
        "--encoder_path",
        type=str,
        default="./distillation_output/student_encoder.pt",
        help="Path to pretrained encoder weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./decoder_output",
        help="Output directory for trained decoder",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    args = parser.parse_args()

    main(
        encoder_path=args.encoder_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
    )
