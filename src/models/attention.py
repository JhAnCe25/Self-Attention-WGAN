import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """SAGAN-style self-attention for 2D feature maps."""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.query = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.key = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, mid)
        k = self.key(x).view(B, -1, N)                      # (B, mid, N)
        v = self.value(x).view(B, C, N)                     # (B, C, N)

        attn = torch.softmax(torch.bmm(q, k) / (q.shape[-1] ** 0.5), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x
