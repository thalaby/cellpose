from vit_sam import Transformer
from vit_tiny import SAMStyleTinyViTEncoder, SAMStyleTinyViTDecoder
from dataset_utils import get_train_val_dataset, get_test_dataset
from settings import MODEL_PATH, TRAINING_ARGS

from train_utils import (
    DistillationModel,
    DistillationTrainer,
    TransformerWrapper,
    StudentSegmentationModel,
    CellposeCustomModel,
)
from torch import nn
from safetensors.torch import load_file
from transformers import TrainingArguments
from logging import getLogger
import torch
import logging
import numpy as np
import wandb

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = getLogger(__name__)



def create_cellpose_model(student_encoder, student_decoder, device="cuda"):
    """Helper function to create a CellposeCustomModel from student encoder/decoder"""
    tiny_network = StudentSegmentationModel(student_encoder, student_decoder).to(device)
    model = CellposeCustomModel(gpu=True, nchan=3, use_bfloat16=False, custom_net=tiny_network)
    return model


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
        "num_train_epochs": 4,
        "per_device_train_batch_size": 128,
        "per_device_eval_batch_size": 128,
        "learning_rate": 5e-4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 10,
        "save_steps": 5000,
        "save_total_limit": 3,
        "seed": 42,
        "fp16": False,  # Disable FP16 since models are already float16
        "gradient_accumulation_steps": 1,
        "remove_unused_columns": False,
        "report_to": "wandb",
        "dataloader_num_workers": 8,
        # "eval_strategy": "epoch",  # Run validation after every epoch
        "save_strategy": "epoch",  # Save checkpoint after every epoch
    }
    default_args.update(kwargs)
    return TrainingArguments(**default_args)


def main(
    output_dir="./distillation_output",
    batch_size=TRAINING_ARGS["train_batch_size"],
):
    """Main training function"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    logger.info(f"Using device: {device}, dtype: {dtype}")
    # Load encoders
    logger.info("Loading student and teacher encoders...")
    student_encoder, teacher_encoder, student_decoder, teacher_model = load_models(
        device=device, dtype=dtype
    )

    # Create distillation model
    logger.info("Creating distillation model...")
    logger.info("Loading student encoder state dict from previous SA1B training...")
    # student_encoder.load_state_dict(torch.load('distillation_output/student_encoder.pt'))
    distillation_model = DistillationModel(student_encoder, teacher_encoder)
    distillation_model.to(device, dtype=dtype)
    state_dict = load_file('good_checkpoints/checkpoint-post-sa-1b/model.safetensors', device='cpu')
    distillation_model.load_state_dict(state_dict)
    distillation_model.to(device, dtype=dtype)

    # Load dataset
    
    wandb.init(project="cellpose_distillation", name="distillation_run")
    
    # Create test dataset from multiple paths
    logger.info("Creating test dataset...")
    test_dataset = get_test_dataset()
    # logger.info(f"test tiles: {len(test_dataset)}")

    logger.info("Creating training dataset...")
    train_dataset, eval_dataset = get_train_val_dataset()
    logger.info(f"Train tiles: {len(train_dataset)}, Eval tiles: {len(eval_dataset)}")
    
    
    # Create cellpose model for evaluation
    cellpose_model = create_cellpose_model(student_encoder, student_decoder, device=device)
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=output_dir,
    )

    # Create trainer with evaluation capabilities
    logger.info("Creating trainer...")
    trainer = DistillationTrainer(
        model=distillation_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_fn=nn.MSELoss(),
        student_decoder=student_decoder,
        cellpose_model=cellpose_model,
        eval_log_steps=TRAINING_ARGS.get('eval_log_steps', None),  # Log eval metrics every N steps
    )
    if TRAINING_ARGS['train']:
        # Start training
        logger.info("Starting distillation training...")
        trainer.train()

    student_encoder = distillation_model.student_encoder

    # Save the student encoder
    logger.info(f"Saving student encoder to {output_dir}/student_encoder.pt")
    torch.save(distillation_model.student_encoder.state_dict(), f"{output_dir}/student_encoder.pt")
    logger.info("Training completed!")

    # Update cellpose model with trained student encoder
    cellpose_model.net = StudentSegmentationModel(distillation_model.student_encoder, student_decoder).to(device)
    
    # Run final test with segmentation metrics
    logger.info("Starting Test with segmentation metrics on test dataset...")
    test_metrics = trainer.test(test_dataset=test_dataset)
    
    # Save metrics to file
    np.savez(f"{output_dir}/segmentation_metrics.npz", **test_metrics)
    logger.info(f"Segmentation test mean AP: {test_metrics['test_mean_ap']:.4f}")


def main_test():
    """Main function for testing model loading and inference"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    logger.info(f"Using device: {device}, dtype: {dtype}")
    # Load encoders
    logger.info("Loading student and teacher encoders...")
    student_encoder, teacher_encoder, student_decoder, teacher_model = load_models(
        device=device, dtype=dtype
    )

    # Create distillation model
    logger.info("Creating distillation model...")
    logger.info("Loading student encoder state dict from previous  training...")
    # student_encoder.load_state_dict(torch.load('distillation_output/student_encoder.pt'))
    distillation_model = DistillationModel(student_encoder, teacher_encoder)
    distillation_model.to(device, dtype=dtype)
    state_dict = load_file('distillation_output/checkpoint-6040/model.safetensors', device='cpu')
    distillation_model.load_state_dict(state_dict)
    distillation_model = distillation_model.to(device, dtype=dtype)

    cellpose_model_mine = create_cellpose_model(distillation_model.student_encoder, student_decoder, device=device)
    cellpose_model_orig = create_cellpose_model(teacher_encoder, student_decoder, device=device)
    test_dataset = get_test_dataset()

    # Create training arguments
    training_args = create_training_args(
        output_dir='test_output',
    )
    # Create trainer with evaluation capabilities
    logger.info("Creating trainer...")
    trainer_mine = DistillationTrainer(
        model=distillation_model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        loss_fn=nn.MSELoss(),
        student_decoder=student_decoder,
        cellpose_model=cellpose_model_mine,
        # eval_log_steps=TRAINING_ARGS.get('eval_log_steps', None),  # Log eval metrics every N steps
    )

    trainer_not_mine = DistillationTrainer(
        model=distillation_model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        loss_fn=nn.MSELoss(),
        student_decoder=student_decoder,
        cellpose_model=cellpose_model_orig,
        # eval_log_steps=TRAINING_ARGS.get('eval_log_steps', None),  # Log eval metrics every N steps
    )

    # Run final test with segmentation metrics
    logger.info("Starting Test with segmentation metrics on test dataset...")
    wandb.init(project="cellpose_distillation", name="distillation_test_run")
    test_metrics_mine = trainer_mine.test(test_dataset=test_dataset)
    test_metrics_not_mine = trainer_not_mine.test(test_dataset=test_dataset)
    import ipdb; ipdb.set_trace()


    


if __name__ == "__main__":
    main_test()
