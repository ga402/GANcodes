
import torch
# import torch.nn as nn
import torch.nn.parallel
# import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torchvision.utils import save_image
# import numpy as np
# from pathlib import Path
import pathlib as pathlib





def train_discriminator(gen, disc, disc_optimiser, images, criterion, batch_size, z_dim, device):
    disc.train() # set train mode...

    # batch_size = images.shape [0]
    # Loss for real images
    #images = images.to(device)
    real_score = disc(images).reshape(-1)
    disc_real_loss = criterion(real_score, torch.ones_like(real_score))

    # Loss for fake images
    z = torch.randn(batch_size, z_dim, 1, 1).to(device)
    fake_images = gen(z)
    fake_score = disc(fake_images).reshape(-1)
    disc_fake_loss = criterion(fake_score, torch.zeros_like(fake_score))

    # Combine losses
    disc_loss = (disc_real_loss + disc_fake_loss) / 2

    # Reset gradients
    disc.zero_grad()

    # Compute gradients
    disc_loss.backward(retain_graph=True)

    # Adjust the parameters using backprop
    disc_optimiser.step()

    return disc_loss, real_score, fake_score


def train_generator(gen, disc, gen_optimiser, criterion, batch_size, z_dim, device):
    gen.train() # set train mode...

    # Generate fake images and calculate loss
    z = torch.randn(batch_size, z_dim, 1, 1).to(device)
    fake_images = gen(z)
    output = disc(fake_images).reshape(-1)
    gen_loss = criterion(output, torch.ones_like(output))

    # Backprop and optimize
    gen.zero_grad()
    gen_loss.backward(retain_graph=True)
    gen_optimiser.step()

    return gen_loss, fake_images




def save_fake_images(index, gen, z_dim, device, save_dir):
    assert isinstance(save_dir, pathlib.PurePath), "save_dir is not a pathlib object"
    fixed_noise = torch.randn(25, z_dim, 1, 1).to(device)
    fake = gen(fixed_noise)

    fake_fname = "fake_images-{0:0=4d}.png".format(index)
    print("Saving", fake_fname)
    save_image(fake, save_dir / fake_fname, nrow=5)


def fit(
    disc,
    gen,
    disc_optimiser,
    gen_optimiser,
    dataloader,
    criterion,
    num_epochs,
    batch_size,
    z_dim,
    device,
    save_dir,
    save_fn=None,
):

    step = 0

    d_losses, g_losses, real_scores, fake_scores = [], [], [], []
    total_step = len(dataloader)

    for epoch in range(num_epochs):
        for i, images in enumerate(dataloader, 0):

            # Train the descriminator and generator
            images = images[0].to(device)
            disc_loss, real_score, fake_score = train_discriminator(
                gen, disc, disc_optimiser, images, criterion, batch_size, z_dim, device
            )
            gen_loss, fake_images = train_generator(
                gen, disc, gen_optimiser, criterion, batch_size, z_dim, device
            )

            # Inspect the losses
            if (i + 1) % 10 == 0:
                d_losses.append(disc_loss.item())
                g_losses.append(gen_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print(
                    "Epoch [{}/{}], Step [{}/{}], disc_loss: {:.4f}, gen_loss: {:.4f}, D(x): {:.2f}, D (G(z)): {:.2f}".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        total_step,
                        disc_loss.item(),
                        gen_loss.item(),
                        real_score.mean().item(),
                        fake_score.mean().item(),
                    )
                )

                step += 1

        # Sample and save images
        if (save_fn is not None) and ((epoch + 1) % 10 == 0):
            save_fn(epoch + 1, gen, z_dim, device, save_dir)

    return dict(
        disc_loss=d_losses,
        gen_loss=g_losses,
        real_score=real_scores,
        fake_score=fake_scores,
    )
