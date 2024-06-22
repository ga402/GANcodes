import argparse
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import imageio
import os
import yaml
from json import dump as json_dump
from utils.general import *
from utils.data_loader import *
from models import *





def get_program_parameters():
    description = "2D DCGAN"
    epilogue = """
    This is DCGAN script for 2 images
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilogue,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataroot", help="Root directory for dataset")
    parser.add_argument("--label_data", type=str, help="label data.csv")
    parser.add_argument("--workers", help="Number of workers for dataloader")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size during training"
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
    #torch.use_deterministic_algorithms(True)

    args = get_program_parameters()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.yaml", "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('torch version:',torch.__version__)
	print('device:', device)




	# Model
	D_output_dim = 1
	num_filters = [1024, 512, 256, 128]
	class_list = ['baseline', 'dense_cluster', 'loose_cluster']
	class_num = len(class_list)

	# dataroot
    dataroot = args.dataroot

    # Number of workers for dataloader
    workers = int(args.workers)

    # Batch size during training
    batch_size = int(args.batch_size)

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    img_size = int(args.image_size)

    # Number of channels in the training images. For color images this is 3
    nc = int(args.nc)

    # Size of z latent vector (i.e. size of generator input)
    z_size = int(args.nz)

    # Number of training epochs
    epochs = int(args.num_epochs)

    # Learning rate for optimizers
    learning_rate = float(args.lr)

    # Beta1 hyperparameter for Adam optimizers
    beta1 = float(args.beta1)
    #betas = (0.5, 0.999)

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = int(args.ngpu)
    
    # number of iterations before training the generator...
    n_critic = int(args.n_critic)


    # load the data
    dataset = BMData(args.label_data, args.dataroot,transform=transform)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=workers)


	generator = Generator(z_size, class_num, num_filters, nc).to(device)
	discriminator = Discriminator(nc, class_num, num_filters[::-1], D_output_dim).to(device)



	# Handle multi-GPU if desired
    if (device.type == "cuda") and (ngpu > 1):
        discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
        generator = nn.DataParallel(generator, list(range(ngpu)))


	print(discriminator)

	print(generator)

	# testing models
    print("testing models")
    testm()




    #
    # # Optimizer
	# Loss function
	criterion = torch.nn.BCELoss()

	# Optimizers
	g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
	d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))


	# label preprocessing
	onehot = torch.zeros(class_num, class_num)
	onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2]).view(class_num, 1), 1).view(class_num, class_num, 1, 1)
	fill = torch.zeros([class_num, class_num, img_size, img_size])
	for i in range(class_num):
	    fill[i, i, :, :] = 1



	# generator a set of sample images
	temp_noise0_ = torch.randn(4, z_size)
	temp_noise0_ = torch.cat([temp_noise0_, temp_noise0_], 0)
	temp_noise1_ = torch.randn(4, z_size)
	temp_noise1_ = torch.cat([temp_noise1_, temp_noise1_], 0)

	fixed_noise = torch.cat([temp_noise0_, temp_noise1_], 0)
	fixed_label = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(torch.LongTensor).squeeze()

	fixed_noise = fixed_noise.view(-1, z_size, 1, 1)
	fixed_label = onehot[fixed_label]




	# Training GAN
	D_avg_losses = []
	G_avg_losses = []

	step = 0
	for epoch in range(epochs):
	    D_losses = []
	    G_losses = []

	    if epoch == 5 or epoch == 10:
	        g_optimizer.param_groups[0]['lr'] /= 10
	        d_optimizer.param_groups[0]['lr'] /= 10

	    # minibatch training
	    for i, (images, labels) in enumerate(data_loader):

	        # image data
	        mini_batch = images.size()[0]
	        x_ = Variable(images.cuda())

	        # labels
	        y_real_ = Variable(torch.ones(mini_batch).cuda())
	        y_fake_ = Variable(torch.zeros(mini_batch).cuda())
	    
	        # labels of size (batch_size, in_channels, H, W)
	        c_fill_ = Variable(fill[labels].cuda())

	        # Train discriminator with real data
	        D_real_decision = discriminator(x_, c_fill_).squeeze()
	        D_real_loss = criterion(D_real_decision, y_real_)

	        # Train discriminator with fake data
	        z_ = torch.randn(mini_batch, z_size).view(-1, z_size, 1, 1)
	        z_ = Variable(z_.cuda())

	        c_ = (torch.rand(mini_batch, 1) * class_num).type(torch.LongTensor).squeeze()
	        c_onehot_ = Variable(onehot[c_].cuda())
	        gen_image = generator(z_, c_onehot_)

	        c_fill_ = Variable(fill[c_].cuda())
	        D_fake_decision = discriminator(gen_image, c_fill_).squeeze()
	        D_fake_loss = criterion(D_fake_decision, y_fake_)
	  
	        # Back propagation
	        D_loss = D_real_loss + D_fake_loss
	        discriminator.zero_grad()
	        D_loss.backward()
	        d_optimizer.step()

	        # Train generator
	        z_ = torch.randn(mini_batch, z_size).view(-1, z_size, 1, 1)
	        z_ = Variable(z_.cuda())

	        c_ = (torch.rand(mini_batch, 1) * class_num).type(torch.LongTensor).squeeze()
	        c_onehot_ = Variable(onehot[c_].cuda())
	        gen_image = generator(z_, c_onehot_)

	        c_fill_ = Variable(fill[c_].cuda())
	        D_fake_decision = discriminator(gen_image, c_fill_).squeeze()
	        G_loss = criterion(D_fake_decision, y_real_)

	        # Back propagation
	        generator.zero_grad()
	        G_loss.backward()
	        g_optimizer.step()

	        # loss values
	        D_losses.append(D_loss.item())
	        G_losses.append(G_loss.item())

	        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
	              % (epoch+1, epochs, i+1, len(data_loader), D_loss.item(), G_loss.item()))
	    
	    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
	    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

	    # avg loss values for plot
	    D_avg_losses.append(D_avg_loss)
	    G_avg_losses.append(G_avg_loss)
	       
	    plot_loss(D_avg_losses, G_avg_losses, epoch, args.loss_plot, save=True, save_dir=save_dir)

	    # Show result for fixed noise
	    plot_result(generator, fixed_noise, fixed_label, epoch,, args.prediction_image_plot, save=True, save_dir=save_dir)

        # save state
    torch.save(generator.state_dict(), save_dir / "generator.pth")
    torch.save(discriminator.state_dict(), save_dir / "discriminator.pth")

    history = {'d_loss':D_avg_losses, 'g_loss': G_avg_losses}

    with open(save_dir / 'history.json', 'w', encoding='utf-8') as f:
        json_dump(history, f, ensure_ascii=False, indent=4)

	plot_morph_result(generator, epoch, args.prediction_image_plot, save=True, save_dir=save_dir)








