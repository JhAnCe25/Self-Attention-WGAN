import torch


def compute_gradient_penalty(critic, real_imgs, fake_imgs, device):
    """Gradient penalty for WGAN-GP (Gulrajani et al., 2017).

    Enforces the 1-Lipschitz constraint on the critic by penalising
    gradients that deviate from unit norm, evaluated on interpolated
    samples between real and generated distributions.

    Args:
        critic:    the Critic network
        real_imgs: batch of real images, shape (N, C, H, W)
        fake_imgs: batch of fake images, shape (N, C, H, W)
        device:    torch device

    Returns:
        Scalar gradient penalty (before multiplying by lambda_gp).
    """
    batch_size = real_imgs.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1.0 - alpha) * fake_imgs).requires_grad_(True)

    d_interp = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = (torch.relu(gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gradient_penalty
