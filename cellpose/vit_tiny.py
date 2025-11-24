from torch import nn
import torch
import torch.nn.functional as F
from mobile_sam import sam_model_registry
import time


class SAMStyleTinyViTEncoder(nn.Module):
    def __init__(
        self,
        device="cuda",
        dtype=torch.bfloat16,
        in_chans: int = 3,
        image_size: int = 256,
        ps: int = 8,
        **_
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_chans = in_chans
        self.image_size = image_size
        self.ps = ps

        # Load MobileSAM and grab the image encoder
        sam = sam_model_registry["custom_vit_t"]().to(
            device=device, dtype=dtype
        )
        self.encoder = sam.image_encoder

    def forward(self, x):
        """Encoder: patch embedding + transformer layers + neck"""
        x = self.encoder.patch_embed(x)
        
        for layer in self.encoder.layers:
            x = layer(x)

        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Token count {N} is not a square number"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.encoder.neck(x)
        
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        return x


class SAMStyleTinyViTDecoder(nn.Module):
    def __init__(self, ps: int = 8, nout: int = 3, **_):
        super().__init__()
        self.ps = ps
        self.nout = nout
        self.out = nn.Conv2d(256, self.nout * ps**2, kernel_size=1)
        self.W2 = nn.Parameter(torch.eye(self.nout * ps**2).reshape(self.nout*ps**2, self.nout, ps, ps), 
                               requires_grad=False)

    def forward(self, x):
        """Decoder: readout and reconstruction"""
        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride=self.ps, padding=0)
        return x1


class SAMStyleTinyViT(nn.Module):
    def __init__(
        self,
        device="cuda",
        dtype=torch.bfloat16,
        in_chans: int = 3,
        image_size: int = 256,
        ps: int = 8,
        nout: int = 3,
        **_
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dummy = torch.zeros((1, 256), device=device, dtype=dtype)
        
        self.encoder = SAMStyleTinyViTEncoder(device=device, dtype=dtype, in_chans=in_chans, 
                                             image_size=image_size, ps=ps)
        self.decoder = SAMStyleTinyViTDecoder(ps=ps, nout=nout)

    def forward(self, x):
        neck = self.encoder(x)
        x1 = self.decoder(neck)
        return x1, self.dummy