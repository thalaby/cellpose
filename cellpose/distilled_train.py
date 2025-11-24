from vit_sam import Transformer
from vit_tiny import SAMStyleTinyViTEncoder, SAMStyleTinyViTDecoder
from sa import SA1BDataset
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from logging import getLogger
import logging
import numpy as np

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = getLogger(__name__)
MODEL_PATH = "/storage/timorhalabi/Research/Data/current/model/cellpose_iteration1_anisotropy"


class TransformerWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.neck(x.permute(0, 3, 1, 2))
        return x


class DistillationDatasetWrapper(Dataset):
    """Wrapper for SA1BDataset to work with HuggingFace Trainer
    
    Divides images into 256x256 tiles to handle variable image sizes.
    """
    
    def __init__(self, sa1b_dataset, tile_size=256, dtype=torch.float16):
        self.dataset = sa1b_dataset
        self.tile_size = tile_size
        self.dtype = dtype
        self.tiles = []
        
        # Pre-compute all tiles for all images
        self._create_tiles()
    
    def _create_tiles(self):
        """Create 256x256 tiles from all images in the dataset"""
        logger.info(f"Creating {self.tile_size}x{self.tile_size} tiles from dataset...")
        
        for idx in range(len(self.dataset)):
            img, mask, class_ids = self.dataset[idx]
            img_array = np.array(img)  # (H, W, C)
            height, width = img_array.shape[:2]
            
            # Extract tiles
            for y in range(0, height, self.tile_size):
                for x in range(0, width, self.tile_size):
                    # Get tile boundaries (with padding if necessary)
                    y_end = min(y + self.tile_size, height)
                    x_end = min(x + self.tile_size, width)
                    
                    tile = img_array[y:y_end, x:x_end, :]
                    
                    # Pad tile if it's smaller than tile_size x tile_size
                    if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                        padded_tile = np.zeros((self.tile_size, self.tile_size, tile.shape[2]), 
                                              dtype=tile.dtype)
                        padded_tile[:tile.shape[0], :tile.shape[1], :] = tile
                        tile = padded_tile
                    
                    self.tiles.append(tile)
        
        logger.info(f"Created {len(self.tiles)} tiles from dataset")
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile = self.tiles[idx]
        
        # Normalize to [0, 1]
        tile = tile.astype(np.float32) / 255.0
        # Convert to tensor and permute to (C, H, W)
        img_tensor = torch.from_numpy(tile).permute(2, 0, 1)
        # Convert to the model's dtype (float16)
        img_tensor = img_tensor.to(self.dtype)
        
        return {"pixel_values": img_tensor}


class DistillationModel(nn.Module):
    """Student model for knowledge distillation"""
    
    def __init__(self, student_encoder, teacher_encoder):
        super().__init__()
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        # Freeze teacher encoder
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values):
        # Student forward pass
        student_neck = self.student_encoder(pixel_values)
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_neck = self.teacher_encoder(pixel_values)
        
        return {
            "student_neck": student_neck,
            "teacher_neck": teacher_neck
        }


class TinyVitEvalWrapper(nn.Module):
    """Wrapper to freeze encoder parameters during training"""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            x = self.decoder(x)
        return x


class DistillationTrainer(Trainer):
    """Custom Trainer for knowledge distillation"""
    
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn or nn.MSELoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs["pixel_values"])
        
        student_neck = outputs["student_neck"]
        teacher_neck = outputs["teacher_neck"]
        
        # Calculate MSE loss between student and teacher neck outputs
        loss = self.loss_fn(student_neck, teacher_neck)
        
        return (loss, outputs) if return_outputs else loss


def load_models(device='cuda', dtype=torch.float16):
    """Load student and teacher encoders"""
    student_encoder = SAMStyleTinyViTEncoder(device=device, dtype=dtype)
    student_decoder = SAMStyleTinyViTDecoder(device=device, dtype=dtype)
    teacher_model = Transformer(backbone="vit_l", ps=8, nout=3, bsize=256, rdrop=0.4, dtype=dtype)
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
):
    """Main training function"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    logger.info(f"Using device: {device}, dtype: {dtype}")
    
    # Load encoders
    logger.info("Loading student and teacher encoders...")
    student_encoder, teacher_encoder, student_decoder = load_models(device=device, dtype=dtype)
    
    # Create distillation model
    logger.info("Creating distillation model...")
    distillation_model = DistillationModel(student_encoder, teacher_encoder)
    distillation_model.to(device, dtype=dtype)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir}...")
    sa1b_dataset = SA1BDataset(dataset_dir=dataset_dir)
    logger.info(f"Dataset loaded with {len(sa1b_dataset)} samples.")
    
    # Wrap dataset for HuggingFace Trainer
    train_dataset = DistillationDatasetWrapper(sa1b_dataset, dtype=dtype)
    
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
    
    # Save the student encoder
    logger.info(f"Saving student encoder to {output_dir}/student_encoder.pt")
    torch.save(student_encoder.state_dict(), f"{output_dir}/student_encoder.pt")

    logger.info("Starting Evaluation..")
    student_eval_model = TinyVitEvalWrapper(student_encoder, student_decoder)
    student_eval_model.to(device, dtype=dtype)
    trainer.model = student_eval_model
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    
    logger.info("Training completed!")


if __name__ == "__main__":
    import numpy as np
    main()