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
            inputs: dict with 'pixel_values' (images), 'labels' (masks), and optionally 'flows' (precomputed)
        """

        pixel_values = inputs["pixel_values"]
        # masks = inputs["labels"]  # (B, 1, H, W)
        # Forward pass through model
        student_outputs = model(pixel_values)  # outputs: (B, 3, H, W) - flows + cellprob
        with torch.no_grad():
            # teacher_model may return a tensor or a tuple/list where the first
            # element is the tensor. Keep this robust.
            teacher_raw = self.teacher_model(pixel_values)
            teacher_outputs = teacher_raw[0] if isinstance(teacher_raw, (list, tuple)) else teacher_raw
        # batch_size = masks.shape[0]
        device = student_outputs.device

        # Check if precomputed flows are available
        # if "flows" in inputs and inputs["flows"] is not None:
        #     # Use precomputed flows (B, 4, H, W): [labels, cellprob, dY, dX]
        #     precomputed_flows = inputs["flows"].to(device)
        #     # Extract [cellprob, dY, dX] - skip labels channel
        #     flows_gt_batch = precomputed_flows[:, 1:, :, :]  # (B, 3, H, W)
        # else:
        #     # Fall back to computing flows on-the-fly
        #     flows_gt = []
        #     for i in range(batch_size):
        #         mask_np = masks[i, 0].cpu().numpy()  # (H, W)

        #         # Convert mask to flows using Cellpose dynamics
        #         # labels_to_flows returns: (4, H, W): [labels, cellprob, dY, dX]
        #         flows = dynamics.labels_to_flows([mask_np])
        #         # flows is a list with one element: (4, H, W) array
        #         flow = flows[0][1:]  # (3, H, W): [cellprob, dY, dX]
        #         flows_gt.append(torch.from_numpy(flow).float())

        #     # Stack flows into batch tensor
        #     flows_gt_batch = torch.stack(flows_gt).to(device)  # (B, 3, H, W)

        # Compute Cellpose segmentation loss
        # train._loss_fn_seg expects: (lbl, y) where lbl is ground truth flows and y is predictions
        # Expected lbl shape (B, 4, H, W) with channel order [labels, cellprob, dY, dX]
        # Student outputs are expected as (B, 3, H, W) in order [dY, dX, cellprob]

        # Move teacher outputs to the same device/dtype as student outputs
        teacher_outputs = teacher_outputs.to(device)

        # Normalize dimensionality: if teacher outputs are (B, H, W, C) convert to (B, C, H, W)
        if teacher_outputs.ndim == 4 and teacher_outputs.shape[-1] in (3, 4):
            # (B, H, W, C) -> (B, C, H, W)
            teacher_outputs = teacher_outputs.permute(0, 3, 1, 2)

        if teacher_outputs.ndim != 4:
            raise ValueError(f"Unexpected teacher output shape: {teacher_outputs.shape}")

        C = teacher_outputs.shape[1]
        if C == 4:
            # assume order already [labels, cellprob, dY, dX]
            flows_gt_batch = teacher_outputs
        elif C == 3:
            # assume teacher gives [dY, dX, cellprob] (same as student)
            # convert to [labels=zeros, cellprob, dY, dX]
            b, _, h, w = teacher_outputs.shape
            zeros = torch.zeros((b, 1, h, w), device=device, dtype=teacher_outputs.dtype)
            cellprob = teacher_outputs[:, 2:3, :, :]
            dy_dx = teacher_outputs[:, 0:2, :, :]
            flows_gt_batch = torch.cat([zeros, cellprob, dy_dx], dim=1)
        else:
            raise ValueError(f"Teacher outputs have unsupported channel count: {C}")

        loss = train._loss_fn_seg(flows_gt_batch, student_outputs, device)

        return (loss, {"predictions": student_outputs}) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """
        Custom evaluation loop that computes validation loss.
        """

        self.model.eval()
        device = self.args.device

        total_loss = 0.0
        num_batches = 0

        logger.info(f"Running {metric_key_prefix} with Cellpose segmentation loss...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=description):
                pixel_values = batch["pixel_values"].to(device)
                masks = batch["labels"].to(device)

                # Forward pass
                outputs = self.model(pixel_values)

                batch_size = masks.shape[0]

                # Convert masks to flows
                flows_gt = []
                for i in range(batch_size):
                    mask_np = masks[i, 0].cpu().numpy()
                    flows = dynamics.labels_to_flows([mask_np])
                    flow = flows[0]
                    flows_gt.append(torch.from_numpy(flow).float())

                flows_gt_batch = torch.stack(flows_gt).to(device)

                # Compute loss
                loss = train._loss_fn_seg(flows_gt_batch, outputs, device)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {f"{metric_key_prefix}_loss": avg_loss}

        logger.info(f"{metric_key_prefix.capitalize()} final loss: {avg_loss:.6f}")

        self.model.train()

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(dataloader.dataset),
        )

    def test(
        self,
        test_dataset=None,
        metric_key_prefix="test",
        log_every_n=10,
    ):
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
            print(
                f"Test dataset: {dataset_name}, samples: {len(test_dataset[dataset_name])}"
            )

            print(f"\nRunning segmentation test on {len(test_dataset)} samples...")

            sample_count = 0
            threshold = [0.5, 0.75, 0.9]
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

                    # # Log intermediate metrics every n samples
                    if sample_count % log_every_n == 0 and self.args.report_to and "wandb" in self.args.report_to:
                        import wandb
                        # Compute metrics on accumulated samples so far
                        ap_partial, tp_partial, fp_partial, fn_partial = metrics.average_precision(
                            masks_gt_all, masks_pred_all
                        )
                        for i, thr in enumerate(threshold):
                            wandb.log({
                                f"{metric_key_prefix}_{thr}_ap_at_{dataset_name}": ap_partial[i],
                                f"{metric_key_prefix}_{thr}_tp_at_{dataset_name}": tp_partial[i],
                                f"{metric_key_prefix}_{thr}_fp_at_{dataset_name}": fp_partial[i],
                                f"{metric_key_prefix}_{thr}_fn_at_{dataset_name}": fn_partial[i],
                                f"{metric_key_prefix}_samples_evaluated": sample_count,
                            })

        # Compute final metrics
        ap, tp, fp, fn = metrics.average_precision(masks_gt_all, masks_pred_all)
        test_metrics = {}
        for i, thr in enumerate(threshold):
            test_metrics[i] = {
                f"{metric_key_prefix}_{thr}_mean_ap": ap[:, i].mean(),
                f"{metric_key_prefix}_{thr}_mean_tp": tp[:, i].mean(),
                f"{metric_key_prefix}_{thr}_mean_fp": fp[:, i].mean(),
                f"{metric_key_prefix}_{thr}_mean_fn": fn[:, i].mean(),
                f"{metric_key_prefix}_total_samples": sample_count,
            }

        # Log final metrics to wandb
        if self.args.report_to and "wandb" in self.args.report_to:
            import wandb

            for i, thr in enumerate(threshold):
                wandb.log(test_metrics[i])

        # Print results
        print(f"\nTest Results ({sample_count} samples):")
        print(f"  Mean AP: {ap[:, 0].mean():.4f}")
        print(f"  Mean TP: {tp[:, 0].mean():.4f}")
        print(f"  Mean FP: {fp[:, 0].mean():.4f}")
        print(f"  Mean FN: {fn[:, 0].mean():.4f}")

        return test_metrics


class StudentSegmentationModelDecoderTrain(nn.Module):
    def __init__(self, encoder, decoder, device="cuda", dtype=torch.float32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dtype = dtype

    def forward(self, x):
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
