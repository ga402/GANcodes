import argparse
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from utils.general import increment_path
from utils.model_utils import save_fake_images# , fit
from utils.wgan_utils import w_fit as fit
from models import *
import yaml
from json import dump as json_dump



def get_program_parameters():
    description = "2D DBGAN"
    epilogue = """
    This is DBGAN script for 2 images
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilogue,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataroot", help="Root directory for dataset")
    parser.add_argument("--workers", help="Number of workers for dataloader")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size during training"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Spatial size of training images. All images will be resized to this size using a transformer",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=3,
        help="Number of channels in the training images. For color images this is 3",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=100,
        help="Size of z latent vector (i.e. size of generator input)",
    )
    parser.add_argument(
        "--ngf", type=int, default=64, help="Size of feature maps in generator"
    )
    parser.add_argument(
        "--ndf", type=int, default=64, help="Size of feature maps in discriminator"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate for optimizers"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Beta1 hyperparameter for Adam optimizers",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="Number of GPUs available. Use 0 for CPU mode.",
    )
    parser.add_argument(
        "--training_image_plot",
        type=str,
        default="training_image_plot.jpg",
        help="plot of the training days",
    )
    parser.add_argument(
        "--loss_plot", type=str, default="loss_plot.jpg", help="plot of loss"
    )
    parser.add_argument(
        "--training_data", type=str, default="training_date.npz", help="loss data"
    )
    parser.add_argument(
        "--score_plot", type=str, default="score_plot.jpg", help="score_plot"
    )
    parser.add_argument(
        "--prediction_image_plot",
        type=str,
        default="prediction_image_plot.jpg",
        help="fake_image_plot",
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--project", default="DBGAN2d", help="save to project/name")
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--accessory_info", default="MODEL DBGAN2d", help="info on the model")
    parser.add_argument("--n_critic", type=int, default=5, help="train generator every n_critic iteration")
    args = parser.parse_args()
    args.save_dir = str(
        increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    )
    return args


if __name__ == "__main__":

    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    args = get_program_parameters()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.yaml", "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    # dataroot
    dataroot = args.dataroot

    # Number of workers for dataloader
    workers = int(args.workers)

    # Batch size during training
    batch_size = int(args.batch_size)

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = int(args.image_size)

    # Number of channels in the training images. For color images this is 3
    nc = int(args.nc)

    # Size of z latent vector (i.e. size of generator input)
    nz = int(args.nz)

    # Size of feature maps in generator
    ngf = int(args.ngf)

    # Size of feature maps in discriminator
    ndf = int(args.ndf)

    # Number of training epochs
    num_epochs = int(args.num_epochs)

    # Learning rate for optimizers
    lr = float(args.lr)

    # Beta1 hyperparameter for Adam optimizers
    beta1 = float(args.beta1)

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = int(args.ngpu)
    
    # number of iterations before training the generator...
    n_critic = int(args.n_critic)

    dataset = dset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    # Decide which device we want to run on
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    )

    # ----------------------------
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=2, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )
    image_path = save_dir / args.training_image_plot
    plt.savefig(image_path)

    # testing models
    print("testing models")
    testm()

    # Create the generator
    netG = Generator(ngpu, nz, nc, ngf).to(device)

    # Create the Discriminator
    netD = Discriminator(ngpu, nc, ndf).to(device)

    # Handle multi-GPU if desired
    if (device.type == "cuda") and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Print the model
    print(" -------- netG ----------")
    print(netG)
    print(" -------- netD ----------")
    print(netD)

    # Initialize the ``BCELoss`` function
    # criterion = nn.BCELoss() - changed to nn.BCELossLogits (19/3/24- first modification)
    
    # criterion = nn.BCEWithLogitsLoss() - not used for the 

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Setup Adam optimizers for both G and D
    optimiserD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimiserG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop
    history = fit(
        netD,
        netG,
        optimiserD,
        optimiserG,
        dataloader,
        n_critic, #criterion - change in w_fit form
        num_epochs,
        batch_size,
        nz,
        device,
        save_dir,
        save_fake_images,
    )

    # save state
    torch.save(netG.state_dict(), save_dir / "netG.pth")
    torch.save(netD.state_dict(), save_dir / "netD.pth")

    # save history
    with open(save_dir / 'history.json', 'w', encoding='utf-8') as f:
        json_dump(history, f, ensure_ascii=False, indent=4)


    try:
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(history["disc_loss"], "-")
        plt.plot(history["gen_loss"], "-")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["Discriminator", "Generator"])
        image_path = save_dir / args.loss_plot
        plt.savefig(image_path)
    except:
        print("unable to save loss figures")

    try:
        plt.figure(figsize=(10, 5))
        plt.title("Scores")
        plt.plot(history["real_score"], "-")
        plt.plot(history["fake_score"], "-")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.legend(["Real Score", "Fake Score"])
        image_path = save_dir / args.score_plot
        plt.savefig(image_path)
    except:
        print("unable to save scores figures")

    # # Lists to keep track of progress
    # img_list = []
    # G_losses = []
    # D_losses = []
    # iters = 0

    # print("Starting Training Loop...")
    # # For each epoch
    # for epoch in range(num_epochs):
    #     # For each batch in the dataloader
    #     for i, data in enumerate(dataloader, 0):

    #         ############################
    #         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    #         ###########################
    #         ## Train with all-real batch
    #         netD.zero_grad()
    #         # Format batch
    #         real_cpu = data[0].to(device)
    #         b_size = real_cpu.size(0)
    #         label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    #         # Forward pass real batch through D
    #         output = netD(real_cpu).view(-1)
    #         # Calculate loss on all-real batch
    #         errD_real = criterion(output, label)
    #         # Calculate gradients for D in backward pass
    #         errD_real.backward()
    #         D_x = output.mean().item()

    #         ## Train with all-fake batch
    #         # Generate batch of latent vectors
    #         noise = torch.randn(b_size, nz, 1, 1, device=device)
    #         # Generate fake image batch with G
    #         fake = netG(noise)
    #         label.fill_(fake_label)
    #         # Classify all fake batch with D
    #         output = netD(fake.detach()).view(-1)
    #         # Calculate D's loss on the all-fake batch
    #         errD_fake = criterion(output, label)
    #         # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    #         errD_fake.backward()
    #         D_G_z1 = output.mean().item()
    #         # Compute error of D as sum over the fake and the real batches
    #         errD = errD_real + errD_fake
    #         # Update D
    #         optimizerD.step()

    #         ############################
    #         # (2) Update G network: maximize log(D(G(z)))
    #         ###########################
    #         netG.zero_grad()
    #         label.fill_(real_label)  # fake labels are real for generator cost
    #         # Since we just updated D, perform another forward pass of all-fake batch through D
    #         output = netD(fake).view(-1)
    #         # Calculate G's loss based on this output
    #         errG = criterion(output, label)
    #         # Calculate gradients for G
    #         errG.backward()
    #         D_G_z2 = output.mean().item()
    #         # Update G
    #         optimizerG.step()

    #         # Output training stats
    #         if i % 50 == 0:
    #             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
    #                   % (epoch, num_epochs, i, len(dataloader),
    #                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    #         # Save Losses for plotting later
    #         G_losses.append(errG.item())
    #         D_losses.append(errD.item())

    #         # Check how the generator is doing by saving G's output on fixed_noise
    #         if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
    #             with torch.no_grad():
    #                 fake = netG(fixed_noise).detach().cpu()
    #             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    #         iters += 1

    # ----

    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses,label="G")
    # plt.plot(D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # image_path = save_dir / args.loss_plot
    # plt.savefig(image_path)

    # Grab a batch of real images from the dataloader
    # real_batch = next(iter(dataloader))

    # Plot the real images
    try:
        plt.figure(figsize=(8, 8))
        fixed_noise = torch.randn(16, 100, 1, 1).to(device)
        fake = netG(fixed_noise)
        # imgs = fake.cpu().detach()
        imgs = vutils.make_grid(fake, nrow=4, normalize=True, padding=0).cpu()
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(imgs, (1, 2, 0)))
        image_path = save_dir / args.prediction_image_plot
        plt.savefig(image_path)
    except:
        print("unable to generate prediction of fake images")
