import torch

from cellpose.Mobile.utils.settings import MODEL_PATH
from cellpose.Mobile.models.train_utils import (CellposeCustomModel,
                         StudentSegmentationModel,
                         TransformerWrapper)

from cellpose.Mobile.models.vit_tiny import SAMStyleTinyViTEncoder, SAMStyleTinyViTDecoder
from cellpose.vit_sam import Transformer

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