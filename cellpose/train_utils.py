from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from cellpose import models
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import os
import re
import tifffile


class StudentSegmentationModel(nn.Module):
    def __init__(self, encoder, decoder, device="cuda", dtype=torch.float32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dtype = dtype
        self.dummy = torch.zeros((1, 256), device=device, dtype=dtype)

    def forward(self, x):
        feat = self.encoder(x)     # neck output
        out = self.decoder(feat)
        return out, self.dummy

class CellposeCustomModel(models.CellposeModel):
    def __init__(self, gpu=True, nchan=3, custom_net=None, use_bfloat16=False):
        super().__init__(gpu=gpu, nchan=nchan, use_bfloat16=use_bfloat16)
        self.net = custom_net if custom_net is not None else self.net


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

        return {"student_neck": student_neck, "teacher_neck": teacher_neck}

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




class DistillationTrainer(Trainer):
    """Custom Trainer for knowledge distillation with segmentation evaluation"""

    def __init__(self, *args, loss_fn=None, student_decoder=None, cellpose_model=None, eval_log_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.student_decoder = student_decoder
        self.cellpose_model = cellpose_model
        self.eval_log_steps = eval_log_steps  # Log intermediate eval metrics every N steps

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(inputs["pixel_values"])

        student_neck = outputs["student_neck"]
        teacher_neck = outputs["teacher_neck"]

        # Calculate MSE loss between student and teacher neck outputs
        loss = self.loss_fn(student_neck, teacher_neck)

        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """
        Custom evaluation loop that computes validation MSE between
        student and teacher necks. Metrics are returned to Trainer,
        which handles wandb logging.
        """
        self.model.eval()
        device = self.args.device

        total_loss = 0.0
        num_batches = 0

        print(f"\nRunning {metric_key_prefix} with MSE loss...")

        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc=description), start=1):
                imgs = batch["pixel_values"].to(device)

                outputs = self.model(imgs)
                student_neck = outputs["student_neck"]
                teacher_neck = outputs["teacher_neck"]

                loss = self.loss_fn(student_neck, teacher_neck)
                total_loss += loss.item()
                num_batches += 1

                # Optional: intermediate logging via Trainer.log()
                if self.eval_log_steps is not None and step % self.eval_log_steps == 0:
                    intermediate_avg_loss = total_loss / self.eval_log_steps
                    print(
                        f"{metric_key_prefix.capitalize()} step {step}: "
                        f"mean loss = {intermediate_avg_loss:.6f}"
                    )
                    # This will be forwarded to wandb with correct global_step

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {f"{metric_key_prefix}_loss": avg_loss}

        print(f"{metric_key_prefix.capitalize()} final MSE loss: {avg_loss:.6f}")

        # DO NOT call wandb.log here. Trainer.evaluate() will call self.log(metrics).
        self.model.train()

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(dataloader.dataset),
        )
    
    def test(self, test_dataset=None, ignore_keys=None, metric_key_prefix="test", log_every_n=10):
        """
        Run final test with segmentation metrics on image-mask pairs.
        This should be called after training is complete.
        Logs intermediate results to wandb every n samples.
        
        Args:
            test_dataset: Dataset with image-mask pairs for segmentation evaluation
            log_every_n: Log metrics to wandb every n samples (default: 10)
        """
        from cellpose import metrics
        
        if test_dataset is None:
            raise ValueError("No test dataset provided")
        
        # Ensure model is in eval mode
        self.model.eval()
        device = self.args.device
        
        masks_gt_all = []
        masks_pred_all = []
        
        # Create dataloader for test
        for dataset_name in test_dataset.keys():
            print(f"Test dataset: {dataset_name}, samples: {len(test_dataset[dataset_name])}")
            test_dataloader = self.get_test_dataloader(test_dataset[dataset_name])
            
            print(f"\nRunning segmentation test on {len(test_dataset)} samples...")
            
            sample_count = 0
            threshold = np.arange(0.5, 1.0, 0.05)
            
            with torch.no_grad():
                for image in tqdm(test_dataset[dataset_name], desc="Testing"):
                    # Extract images and masks from batch dict
                    img = image["pixel_values"].to(device)
                    masks_gt = image["labels"]
                    img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    mask_gt = masks_gt.squeeze().cpu().numpy().astype(np.int16)
                    
                    # Run cellpose evaluation
                    masks_pred, flows, styles = self.cellpose_model.eval(img)
                    
                    masks_gt_all.append(mask_gt)
                    masks_pred_all.append(masks_pred)
                    sample_count += 1
                    
                    # Log intermediate metrics every n samples
                    if sample_count % log_every_n == 0 and self.args.report_to and "wandb" in self.args.report_to:
                        import wandb

                        # Compute metrics on accumulated samples so far
                        ap_partial, tp_partial, fp_partial, fn_partial = metrics.average_precision(
                            masks_gt_all, masks_pred_all, threshold=threshold
                        )
                        
                        wandb.log({
                            f"{metric_key_prefix}_mean_ap_at_{dataset_name}": ap_partial.mean(),
                            f"{metric_key_prefix}_mean_tp_at_{dataset_name}": tp_partial.mean(),
                            f"{metric_key_prefix}_mean_fp_at_{dataset_name}": fp_partial.mean(),
                            f"{metric_key_prefix}_mean_fn_at_{dataset_name}": fn_partial.mean(),
                            f"{metric_key_prefix}_samples_evaluated": sample_count,
                        })
            
        # Compute final metrics
        ap, tp, fp, fn = metrics.average_precision(masks_gt_all, masks_pred_all, threshold=threshold)
        
        # Compute mean metrics
        mean_ap = ap.mean()
        mean_tp = tp.mean()
        mean_fp = fp.mean()
        mean_fn = fn.mean()
        
        # Create metrics dict for logging
        test_metrics = {
            f"{metric_key_prefix}_mean_ap": mean_ap,
            f"{metric_key_prefix}_mean_tp": mean_tp,
            f"{metric_key_prefix}_mean_fp": mean_fp,
            f"{metric_key_prefix}_mean_fn": mean_fn,
            f"{metric_key_prefix}_total_samples": sample_count,
        }
        
        # Log final metrics to wandb
        if self.args.report_to and "wandb" in self.args.report_to:
            import wandb
            wandb.log(test_metrics)
        
        # Print results
        print(f"\nTest Results ({sample_count} samples):")
        print(f"  Mean AP: {mean_ap:.4f}")
        print(f"  Mean TP: {mean_tp:.4f}")
        print(f"  Mean FP: {mean_fp:.4f}")
        print(f"  Mean FN: {mean_fn:.4f}")
        
        return test_metrics