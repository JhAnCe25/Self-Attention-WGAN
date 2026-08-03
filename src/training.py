import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .losses import compute_gradient_penalty
from .models import Critic, Generator, initialize_weights
from .octave_io import generate_and_export, get_latest_epoch

PAD = 14  # noise + pad, to avoid padding with a lot of zeros in the first conv


def train(args, train_dataset):
    """Run WGAN-GP training.

    Note: `train_dataset` is passed explicitly (unlike the original notebook,
    which silently read a `train_dataset` global set up in an earlier cell).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")
    print(f"Dataset size  : {len(train_dataset)}")
    print(f"Channels      : {args.channels_img}")

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    batch_check = next(iter(dataloader))
    print(f"Batch shape   : {batch_check.shape}")

    # --- Models ---------------------------------------------------------
    netC_ = Critic(channels_img=args.channels_img).to(device)
    netG_ = Generator(in_channels=args.noise_channels, out_channels=args.channels_img).to(device)

    netG_.apply(initialize_weights)
    netC_.apply(initialize_weights)

    # --- WGAN-GP optimizers (Adam, beta1=0, beta2=0.9 as per paper) -----
    optimizerC = optim.Adam(netC_.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG_.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    start_epoch = 0
    if args.resume_training:
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        latest_epoch = get_latest_epoch(checkpoint_dir)
        if latest_epoch >= 0:
            print(f"Resuming training from epoch {latest_epoch}")
            netG_.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, f"generator_epoch_{latest_epoch}.pth"), map_location=device)
            )
            netC_.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, f"critic_epoch_{latest_epoch}.pth"), map_location=device)
            )
            start_epoch = latest_epoch + 1
        else:
            print("No checkpoints found. Starting new training.")

    # Fixed noise at several output resolutions, for sample previews each epoch
    fixed_noise_128 = torch.randn(1, args.noise_channels, args.noise_height + PAD, args.noise_width + PAD, device=device)
    fixed_noise_256 = torch.randn(1, args.noise_channels, args.noise_height * 2 + PAD, args.noise_width * 2 + PAD, device=device)
    fixed_noise_512 = torch.randn(1, args.noise_channels, args.noise_height * 4 + PAD, args.noise_width * 4 + PAD, device=device)
    fixed_noise_1024 = torch.randn(1, args.noise_channels, args.noise_height * 8 + PAD, args.noise_width * 8 + PAD, device=device)

    print("\nStarting WGAN-GP training...")
    for epoch in range(start_epoch, args.num_epochs):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # --- Critic steps --------------------------------------------
            for _ in range(args.n_critic):
                noise = torch.randn(
                    batch_size, args.noise_channels, args.noise_height + PAD, args.noise_width + PAD, device=device
                )
                fake_imgs = netG_(noise).detach()

                netC_.zero_grad()

                loss_real = netC_(real_imgs).mean()
                loss_fake = netC_(fake_imgs).mean()
                gp = compute_gradient_penalty(netC_, real_imgs, fake_imgs, device)

                lossC = loss_fake - loss_real + args.lambda_gp * gp
                lossC.backward()
                optimizerC.step()

            # --- Generator step ------------------------------------------
            netG_.zero_grad()
            noise = torch.randn(
                batch_size, args.noise_channels, args.noise_height + PAD, args.noise_width + PAD, device=device
            )
            fake_imgs = netG_(noise)
            lossG = -netC_(fake_imgs).mean()
            lossG.backward()
            optimizerG.step()

            if i % args.log_interval == 0:
                print(
                    f"[Epoch {epoch:03d}/{args.num_epochs}] "
                    f"[Batch {i:03d}/{len(dataloader)}] "
                    f"Loss_C: {lossC.item():.4f}  Loss_G: {lossG.item():.4f}  "
                    f"W_dist: {-lossC.item():.4f}"
                )

        # ── End-of-epoch: save preview samples at multiple resolutions ──
        with torch.no_grad():
            for tag, fixed_noise in [
                ("128", fixed_noise_128),
                ("256", fixed_noise_256),
                ("512", fixed_noise_512),
                ("1024", fixed_noise_1024),
            ]:
                fake = netG_(fixed_noise).detach().cpu()
                fake = (fake + 1) / 2
                print(f"Epoch {epoch:03d} - {tag}x{tag} output shape: {fake.shape}")
                ch0 = fake[0:1, 0:1, :, :]
                ch1 = fake[0:1, 1:2, :, :]
                side = torch.cat([ch0, ch1], dim=3)
                save_image(
                    side,
                    os.path.join(args.output_dir, "samples", f"epoch_{epoch:03d}_{tag}_both_channels.png"),
                    normalize=False,
                )

        # --- Checkpointing ----------------------------------------------
        if epoch % args.save_epoch == 0 or epoch == args.num_epochs - 1:
            ckpt = os.path.join(args.output_dir, "checkpoints")
            torch.save(netG_.state_dict(), os.path.join(ckpt, f"generator_epoch_{epoch}.pth"))
            torch.save(netC_.state_dict(), os.path.join(ckpt, f"critic_epoch_{epoch}.pth"))

        # --- Generate and export to Octave/MATLAB format -----------------
        print(f"Generating and exporting Octave documents for epoch {epoch}...")
        generate_and_export(
            output_dir=args.output_dir,
            noise_channels=args.noise_channels,
            noise_height=args.noise_height + PAD,
            noise_width=args.noise_height + PAD,
            channels_img=args.channels_img,
            sample_size=args.sample_size * 4,
            epoch=epoch,
            gen_model=netG_,
            output_resolution="128",
        )
        generate_and_export(
            output_dir=args.output_dir,
            noise_channels=args.noise_channels,
            noise_height=args.noise_height * 4 + PAD,
            noise_width=args.noise_height * 4 + PAD,
            channels_img=args.channels_img,
            sample_size=args.sample_size * 2,
            epoch=epoch,
            gen_model=netG_,
            output_resolution="512",
        )
        generate_and_export(
            output_dir=args.output_dir,
            noise_channels=args.noise_channels,
            noise_height=args.noise_height * 8 + PAD,
            noise_width=args.noise_height * 8 + PAD,
            channels_img=args.channels_img,
            sample_size=args.sample_size,
            epoch=epoch,
            gen_model=netG_,
            output_resolution="1024",
        )

    print("Training finished.")
    return netG_, netC_
