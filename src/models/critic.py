import torch
import torch.nn as nn

from .attention import SelfAttention


class Critic(nn.Module):
    def __init__(self, channels_img=2, img_size=128):
        super().__init__()
        self.block1 = self._block(channels_img, 32)
        self.block2 = self._block(32, 32)
        self.block3 = self._block(32, 64)
        self.attn = SelfAttention(64, reduction=8)  # ~31x31 feature map
        self.block4 = self._block(64, 64)

        with torch.no_grad():
            dummy = torch.zeros(1, channels_img, img_size, img_size)
            flat = self._forward_conv(dummy).view(1, -1).shape[1]
        print(f"Critic: flattened size = {flat}")

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 1),
        )

    @staticmethod
    def _block(in_ch, out_ch, kernel_size=3, stride=2, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        )

    def _forward_conv(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attn(x)
        x = self.block4(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        return self.fc(x)
