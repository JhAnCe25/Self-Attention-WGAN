import torch.nn as nn

from .attention import SelfAttention


class Generator(nn.Module):
    """WGAN-NoLinV2: noise (N,2,64,64) -> texture (N,2,512,512)."""

    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=15, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.attn0 = SelfAttention(512, reduction=8)  # 64x64, 512ch

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 64 -> 128
            nn.Conv2d(512, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 128 -> 256
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 256 -> 512
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x):
        # x shape: (N, 2, 64, 64) [before padding, see training.py `pad`]
        x = self.initial(x)   # (N, 512, 64, 64)
        x = self.attn0(x)     # (N, 512, 64, 64)
        x = self.up1(x)       # (N, 64, 128, 128)
        x = self.up2(x)       # (N, 32, 256, 256)
        x = self.up3(x)       # (N, out_channels, 512, 512)
        return x
