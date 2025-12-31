import logging
import torch
import numpy as np

from torch import nn
from pathlib import Path
from transformers import Trainer
from tqdm import tqdm
from transformers.trainer_utils import EvalLoopOutput
from cellpose import dynamics, train
from cellpose.Mobile.models.vit_tiny import SAMStyleTinyViTEncoder
from cellpose.Mobile.models.train_utils import perform_test

logger = logging.getLogger(__name__)


class DecoderTrainer(Trainer):
    """Custom Trainer for decoder training with Cellpose segmentation loss"""

    def __init__(
        self,
        *args,
        encoder=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cellpose_model = None
        self.teacher_model = None

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute Cellpose segmentation loss.

        Args:
            model: StudentSegmentationModel (encoder + decoder)
            inputs: dict with 'pixel_values' (images), 'labels' (flows)
        """

        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]  # (B, 3, H, W)
        cellprobs = labels[:, 0:1, :, :]  # (B, 1, H, W)
        flowsY = labels[:, 1:2, :, :]  # (B, 1, H, W)
        flowsX = labels[:, 2:3, :, :]  # (B, 1, H, W)
        flows = torch.cat([cellprobs, flowsY, flowsX], dim=1)

        # Forward pass through model
        student_outputs = model(pixel_values)  # outputs: (B, 3, H, W) - flows + cellprob
        with torch.no_grad():
            # teacher_model may return a tensor or a tuple/list where the first
            # element is the tensor. Keep this robust.
            teacher_raw = self.teacher_model(pixel_values)
            teacher_outputs = teacher_raw[0] if isinstance(teacher_raw, (list, tuple)) else teacher_raw
        # batch_size = masks.shape[0]
        device = student_outputs.device
        # Move teacher outputs to the same device/dtype as student outputs
        teacher_outputs = teacher_outputs.to(device)
        loss = train._loss_fn_seg(flows, student_outputs, device)

        return (loss, {"predictions": student_outputs}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to ensure loss is computed during evaluation.
        """
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            if has_labels:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                loss = loss.detach()
            else:
                loss = None
            
        return (loss, None, None)

    def evaluate_loss(self, eval_dataset=None, batch_size=None):
        """
        Basic evaluation loop that computes the same loss as compute_loss.
        
        Args:
            eval_dataset: Dataset to evaluate on. If None, uses self.eval_dataset
            batch_size: Batch size for evaluation. If None, uses per_device_eval_batch_size
            
        Returns:
            dict: Dictionary containing average loss and number of samples
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}
        
        if batch_size is None:
            batch_size = self.args.per_device_eval_batch_size
        
        # Get device - use model's device if args.device not available
        device = getattr(self.args, 'device', None)
        if device is None:
            device = next(self.model.parameters()).device
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
        )
        
        # Set model to eval mode
        self.model.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        logger.info(f"Running evaluation on {len(eval_dataset)} samples...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute loss using the same logic as compute_loss
                loss = self.compute_loss(self.model, batch, return_outputs=False)
                
                batch_size_actual = batch["pixel_values"].shape[0]
                total_loss += loss.item() * batch_size_actual
                num_samples += batch_size_actual
        
        # Set model back to train mode
        self.model.train()
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        result = {
            "eval_loss": avg_loss,
            "eval_samples": num_samples,
        }
        
        logger.info(f"Evaluation results: {result}")
        
        return result

    def test(
        self,
        test_dataset=None,
        metric_key_prefix="test",
        log_every_n=10,
        model_name="decoder_model",
    ):
        """
        Run final test with segmentation metrics on image-mask pairs.
        This should be called after training is complete.
        Logs intermediate results to wandb every n samples.

        Args:
            test_dataset: Dataset with image-mask pairs for segmentation evaluation
            log_every_n: Log metrics to wandb every n samples (default: 10)
        """
        
        return perform_test(
            model=self.cellpose_model,
            test_dataset=test_dataset,
            metric_key_prefix=metric_key_prefix,
            model_name=model_name,
            report_wandb=self.args.report_to and "wandb" in self.args.report_to,
        )



class StudentSegmentationModelDecoderTrain(nn.Module):
    def __init__(self, encoder, decoder, device="cuda", dtype=torch.float32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dtype = dtype

    def forward(self, x=None, pixel_values=None, **kwargs):
        # Handle both direct input and dict-style input from Trainer
        if x is None:
            x = pixel_values
        if x is None:
            raise ValueError("Either 'x' or 'pixel_values' must be provided")
        
        with torch.no_grad():
            feat = self.encoder(x)  # neck output
        out = self.decoder(feat)
        return out


def load_encoder(encoder_path, device="cuda", dtype=None):
    """Load pretrained encoder weights"""

    if dtype is None:
        dtype = torch.float32

    encoder = SAMStyleTinyViTEncoder(device=device, dtype=dtype)

    if encoder_path and Path(encoder_path).exists():
        logger.info(f"Loading encoder weights from {encoder_path}")
        state_dict = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(state_dict)
        logger.info("Encoder weights loaded successfully")
    else:
        logger.warning(
            f"Encoder path {encoder_path} not found. Using random initialization."
        )

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder
