
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


# def calculate_gradient_penalty(model, real_images, fake_images, device):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake data
#     alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
#     # Get random interpolation between real and fake data
#     interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

#     model_interpolates = model(interpolates)
#     grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

#     # Get gradient w.r.t. interpolates
#     gradients = torch.autograd.grad(
#         outputs=model_interpolates,
#         inputs=interpolates,
#         grad_outputs=grad_outputs,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
#     return gradient_penalty






# def calculate_gradient_penalty(self, real_images, fake_images):
#     eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
#     eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
#     if self.cuda:
#         eta = eta.cuda(self.cuda_index)
#     else:
#         eta = eta

#     interpolated = eta * real_images + ((1 - eta) * fake_images)

#     if self.cuda:
#         interpolated = interpolated.cuda(self.cuda_index)
#     else:
#         interpolated = interpolated

#     # define it to calculate gradient
#     interpolated = Variable(interpolated, requires_grad=True)

#     # calculate probability of interpolated examples
#     prob_interpolated = self.D(interpolated)

#     # calculate gradients of probabilities with respect to examples
#     gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
#                            grad_outputs=torch.ones(
#                                prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
#                                prob_interpolated.size()),
#                            create_graph=True, retain_graph=True)[0]

#     # flatten the gradients to it calculates norm batchwise
#     gradients = gradients.view(gradients.size(0), -1)
    
#     grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
#     return grad_penalty







def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    #alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    alpha = torch.FloatTensor(real_images.size(0),1,1,1).uniform_(0,1)
    alpha = alpha.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3))

    alpha = alpha.to(device)

    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty





























def w_train_discriminator(gen, disc, disc_optimiser, images,  batch_size, z_dim, device):
    disc.train() # set training mode = True

     #batch_size = images.shape [0]
    # Loss for real images
    #images = images.to(device)
    
    # Set discriminator gradients to zero.
    disc.zero_grad()
    
    real_score = disc(images).reshape(-1)
    errD_real = torch.mean(real_score)
    D_x = real_score.mean().item()
    
    # disc_real_loss = criterion(real_score, torch.ones_like(real_score))

    # Loss for fake images
    z = torch.randn(batch_size, z_dim, 1, 1).to(device)
    fake_images = gen(z)
    
    # train with fake...
    fake_score = disc(fake_images).reshape(-1)
    errD_fake = torch.mean(fake_score)
    D_G_z1 = fake_score.mean().item()
    
    gradient_penalty = calculate_gradient_penalty(disc, images.data, fake_images.data, device)
    
    # Add the gradients from the all-real and all-fake batches
    disc_loss = -errD_real + errD_fake + gradient_penalty * 10
    
    #disc_fake_loss = criterion(fake_score, torch.zeros_like(fake_score))

    # Combine losses
    #disc_loss = (disc_real_loss + disc_fake_loss) / 2

    # Reset gradients
    # disc.zero_grad()

    # Compute gradients
    disc_loss.backward(retain_graph=True)

    # Adjust the parameters using backprop
    disc_optimiser.step()

    return disc_loss, real_score, fake_score, D_x, D_G_z1


def w_train_generator(gen, disc, gen_optimiser, batch_size, z_dim, device):
    gen.train()
    
    gen.zero_grad()

    # Generate fake images and calculate loss
    z = torch.randn(batch_size, z_dim, 1, 1).to(device)
    fake_images = gen(z)
    fake_output = disc(fake_images).reshape(-1)
    gen_loss = -torch.mean(fake_output)
    D_G_z2 = fake_output.mean().item()
    # errG.backward(retain_graph=True)
    # gen_loss = criterion(output, torch.ones_like(output))

    # Backprop and optimize
    
    gen_loss.backward()
    gen_optimiser.step()

    return gen_loss, fake_images, D_G_z2


def w_fit(
    disc,
    gen,
    disc_optimiser,
    gen_optimiser,
    dataloader,
    n_critic,
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
            disc_loss, real_score, fake_score, D_x, D_G_z1 = w_train_discriminator(
                gen, disc, disc_optimiser, images, batch_size, z_dim, device
            )
            
            # train the generator every 5th iteration...
            if (i +1 ) % n_critic == 0:
                gen_loss, fake_images, D_G_z2 = w_train_generator(
                    gen, disc, gen_optimiser,  batch_size, z_dim, device
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
                        D_x, #real_score.mean().item(),
                        D_G_z1/D_G_z2 #fake_score.mean().item(),
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

