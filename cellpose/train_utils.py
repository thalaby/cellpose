from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
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

    def __init__(self, *args, loss_fn=None, student_decoder=None, cellpose_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.student_decoder = student_decoder
        self.cellpose_model = cellpose_model

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(inputs["pixel_values"])

        student_neck = outputs["student_neck"]
        teacher_neck = outputs["teacher_neck"]

        # Calculate MSE loss between student and teacher neck outputs
        loss = self.loss_fn(student_neck, teacher_neck)

        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", log_every_n=10):
        """
        Custom evaluation that computes segmentation metrics on image-mask pairs.
        Logs intermediate results to wandb every n samples.
        
        Args:
            log_every_n: Log metrics to wandb every n samples (default: 10)
        """
        from cellpose import metrics
        
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")
        
        # Ensure model is in eval mode
        self.model.eval()
        device = self.args.device
        
        masks_gt_all = []
        masks_pred_all = []
        
        # Create dataloader for evaluation
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        print(f"Running segmentation evaluation on {len(eval_dataset)} samples...")
        
        sample_count = 0
        threshold = np.arange(0.5, 1.0, 0.05)
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Extract images and masks from batch dict
                imgs = batch["pixel_values"].to(device)
                masks_gt = batch["labels"]
                
                # Process each image in batch individually for cellpose
                for i in range(imgs.shape[0]):
                    img = imgs[i].permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    mask_gt = masks_gt[i].squeeze().cpu().numpy().astype(np.int16)
                    
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
                            f"{metric_key_prefix}_mean_ap_at_{sample_count}": ap_partial.mean(),
                            f"{metric_key_prefix}_mean_tp_at_{sample_count}": tp_partial.mean(),
                            f"{metric_key_prefix}_mean_fp_at_{sample_count}": fp_partial.mean(),
                            f"{metric_key_prefix}_mean_fn_at_{sample_count}": fn_partial.mean(),
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
        eval_metrics = {
            f"{metric_key_prefix}_mean_ap": mean_ap,
            f"{metric_key_prefix}_mean_tp": mean_tp,
            f"{metric_key_prefix}_mean_fp": mean_fp,
            f"{metric_key_prefix}_mean_fn": mean_fn,
            f"{metric_key_prefix}_total_samples": sample_count,
        }
        
        # Log final metrics to wandb
        if self.args.report_to and "wandb" in self.args.report_to:
            import wandb
            wandb.log(eval_metrics)
        
        # Print results
        print(f"\nEvaluation Results ({sample_count} samples):")
        print(f"  Mean AP: {mean_ap:.4f}")
        print(f"  Mean TP: {mean_tp:.4f}")
        print(f"  Mean FP: {mean_fp:.4f}")
        print(f"  Mean FN: {mean_fn:.4f}")
        
        return eval_metrics