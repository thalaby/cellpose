import torch
import logging
import os
import numpy as np

from cellpose.Mobile.utils.utils import load_models, create_cellpose_model
from cellpose.Mobile.models.train_utils import DistillationModel, perform_test
from cellpose.Mobile.utils.dataset_utils import get_test_dataset
from safetensors.torch import load_file
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    import wandb

    # Set device and dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    wandb.init(project="cellpose_distillation", name="test_multiple_models")
    model_dir = "./distillation_output"
    # Load pretrained encoder
    logger.info("Loading pretrained encoder...")
    student_encoder, teacher_encoder, student_decoder, teacher_model = load_models(
        device=device, dtype=dtype
    )
    test_dataset = get_test_dataset(partial=True)
    model_dir = "./good_checkpoints"
    # student_encoder.load_state_dict(torch.load('distillation_output/student_encoder.pt'))
    for model_path in ['checkpoint-post-distilled-cell-2712']:#os.listdir(model_dir):
        logger.info(f"Creating distillation model {model_path}...")
        logger.info("Loading student encoder state dict..")
        # if "checkpoint" not in model_path:
        #     continue
        distillation_model = DistillationModel(student_encoder, teacher_encoder)
        distillation_model.to(device, dtype=dtype)
        state_dict = load_file(
            Path(model_dir, model_path, "model.safetensors"), device="cpu"
        )
        distillation_model.load_state_dict(state_dict)
        distillation_model.to(device, dtype=dtype)
        cellpose_model = create_cellpose_model(distillation_model.student_encoder, student_decoder, device=device)
        output = perform_test(
            model=cellpose_model,
            test_dataset=test_dataset,
            metric_key_prefix="final_test",
            model_name=f"distilled_model_{model_path}",
            report_wandb=True,
        )
        del distillation_model
        torch.cuda.empty_cache()
        np.save(f"tests/test_metrics_{model_path}.npy", output)   


if __name__ == "__main__":
    main()