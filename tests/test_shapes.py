import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses import compute_gradient_penalty
from src.models import Critic, Generator, initialize_weights


def test_generator_output_shape():
    gen = Generator(in_channels=2, out_channels=2)
    gen.apply(initialize_weights)
    noise = torch.randn(2, 2, 30, 30)  # noise_height/width (16) + pad (14)
    out = gen(noise)
    # 15x15 conv, no padding, removes exactly the 14px of pad -> back to 16x16,
    # then x8 from three x2 upsampling stages -> 128x128
    assert out.shape == (2, 2, 128, 128)


def test_critic_output_shape():
    critic = Critic(channels_img=2, img_size=128)
    critic.apply(initialize_weights)
    x = torch.randn(4, 2, 128, 128)
    out = critic(x)
    assert out.shape == (4, 1)


def test_gradient_penalty_is_finite():
    critic = Critic(channels_img=2, img_size=128)
    device = torch.device("cpu")
    real = torch.randn(4, 2, 128, 128)
    fake = torch.randn(4, 2, 128, 128)
    gp = compute_gradient_penalty(critic, real, fake, device)
    assert torch.isfinite(gp)
