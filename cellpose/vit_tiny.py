from torch import nn
import torch
import torch.nn.functional as F
from mobile_sam import sam_model_registry
import time


class SAMStyleLongViT(nn.Module):
    def __init__(
        self,
        device="cuda",
        dtype=torch.bfloat16,
        in_chans: int = 3,
        image_size: int = 1024,
        ps: int = 8,
        nout: int = 3,
        **_
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_chans = in_chans
        self.image_size = image_size
        self.ps = ps
        self.nout = nout

        # 1) Load MobileSAM and grab the image encoder
        sam = sam_model_registry["custom_vit_t"]().to(
            device=device, dtype=dtype
        )
        self.encoder = sam.image_encoder  # TinyViT backbone + its neck
        self.dummy = torch.zeros((1, 256), device=device, dtype=dtype)
        self.out = nn.Conv2d(256, self.nout * ps**2, kernel_size=1)

        # W2 reshapes token space to pixel space, not trainable
        self.W2 = nn.Parameter(torch.eye(self.nout * ps**2).reshape(self.nout*ps**2, self.nout, ps, ps), 
                               requires_grad=False)

    def forward(self, x):
        
        # 1. start with image (B,3,H,W)
        t0 = time.time()
        x = self.encoder.patch_embed(x)
        
        # ... whatever the original code does here
        for layer in self.encoder.layers:
            x = layer(x)   # eventually this becomes (B, N, C)

        # 2. NOW x is (B, N, C) → OK to unpack as 3D
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Token count {N} is not a square number"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.encoder.neck(x)
        neck = x
        
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        
        # readout is changed here
        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride=self.ps, padding=0)
        # print(f"total forward: {time.time() - t0:.4f}s")
        
        return x1, self.dummy, neck