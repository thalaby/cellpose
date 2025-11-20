from torch import nn
import torch
import torch.nn.functional as F
from mobile_sam import sam_model_registry


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SAMStyleLongViT(nn.Module):
    def __init__(
        self,
        device="cuda",
        dtype=torch.bfloat16,
        in_chans: int = 3,
        image_size: int = 1024,
        **_
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_chans = in_chans
        self.image_size = image_size

        # 1) Load MobileSAM and grab the image encoder
        sam = sam_model_registry["vit_t"](checkpoint="./weights/mobile_sam.pt").to(
            device=device, dtype=dtype
        )
        self.backbone = sam.image_encoder  # TinyViT backbone + its neck

        # 2) Infer backbone output shape + patch size from a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(
                1, in_chans, image_size, image_size, device=device, dtype=dtype
            )
            feat = self.backbone(dummy)           # (1, C_backbone, Hf, Wf)
        _, C_backbone, Hf, Wf = feat.shape
        patch_size = image_size // Hf            # should be 16 if Hf = 64, image_size=1024

        self.embed_dim = C_backbone              # e.g. 256 for MobileSAM
        self.ps = patch_size                     # e.g. 16
        self.nout = in_chans                     # usually 3

        # 3) SAM-style neck: (C_backbone -> 256 -> 256)
        self.neck = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

        # 4) 1x1 conv that produces patchified channels for deconvolution
        #    out_channels = nout * (ps^2), like in your original code
        self.out = nn.Conv2d(256, self.nout * self.ps**2, kernel_size=1)

        # 5) W2 reshapes token space to pixel space – keep it non-trainable
        W2 = torch.eye(self.nout * self.ps**2).reshape(
            self.nout * self.ps**2, self.nout, self.ps, self.ps
        )
        self.register_buffer("W2", W2)  # buffer instead of Parameter with requires_grad=False

        # 6) Dummy vector to match original interface
        self.dummy = torch.zeros((1, 256), device=device, dtype=dtype)

    def forward(self, x):
        # x: (B, 3, 1024, 1024)
        B = x.shape[0]

        # 1) MobileSAM image encoder: returns feature map, NOT tokens
        #    feat: (B, C_backbone, Hf, Wf) e.g. (B, 256, 64, 64)
        feat = self.backbone(x.to(self.device, dtype=self.dtype))

        # 2) Neck: (B, 256, Hf, Wf)
        feat = self.neck(feat)

        # 3) 1x1 conv to patchified channels: (B, nout * ps^2, Hf, Wf)
        x1 = self.out(feat)

        # 4) De-patchify to pixel space using fixed W2
        #    conv_transpose2d: stride=ps to go back to 1024x1024
        x1 = F.conv_transpose2d(x1, self.W2, stride=self.ps, padding=0)
        # x1: (B, nout, Hf * ps, Wf * ps) -> (B, 3, 1024, 1024) if Hf*ps=1024

        # 5) Expand dummy to batch size
        return x1, self.dummy.expand(B, -1)
