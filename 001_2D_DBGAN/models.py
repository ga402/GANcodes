import torch
import torch.nn as nn
import torch.nn.parallel
#import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
#import torchvision.utils as vutils
# from torchvision.utils import save_image
#import numpy as np
#from pathlib import Path
import pathlib as pathlib

# custom weights initialization called on ``netG`` and ``netD``
# From the DCGAN paper, the authors specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# alternative initialise strategy
def initialise_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# Generator Code


class Generator(nn.Module):
    """a model generates fake images"""

    def __init__(self, ngpu, nz, img_channels, features_g):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # Input: N x z_dim x 1 x 1
        self.main = nn.Sequential(
            self._block(nz, features_g * 32, 4, 2, 0),  # 4x4
            self._block(features_g * 32, features_g * 16, 4, 2, 1),  # 8x8
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 16x16
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 32x32
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 64x64
            nn.ConvTranspose2d(
                features_g * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),  # 128x128
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """a model that judges between real and fake images"""

    def __init__(self, ngpu, img_channels, features_d):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # Input: N x channels_img x 128 x 128
        self.main = nn.Sequential(
            nn.Conv2d(
                img_channels, features_d, kernel_size=4, stride=2, padding=1
            ),  # 64x64
            nn.LeakyReLU(0.2),
            self._block(features_d * 1, features_d * 2, 4, 2, 1),  # 32x32
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0) #,  # 1x1
            # nn.Sigmoid(), - dropped; changed to using BCEwithLogitsLoss as loss function instead. 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        #    out = self.main(x) # updated
        #    out = torch.flatten(out) #updated
        return self.main(x)


def testm():
    """test function to test models"""
    N, in_channels, H, W = 8, 3, 128, 128
    x = torch.randn((N, in_channels, H, W))
    z_dim = 100
    D = Discriminator(1, in_channels, 32)
    initialise_weights(D)
    assert D(x).shape == (N, 1, 1, 1)
    G = Generator(1, z_dim, in_channels, 32)
    initialise_weights(G)
    z = torch.randn((N, z_dim, 1, 1))
    assert G(z).shape == (N, in_channels, H, W)
    print("Success test for 3x128x128 data")





# class Generator(nn.Module):
#    def __init__(self, ngpu, nz, ngf, nc):
#        super(Generator, self).__init__()
#        self.ngpu = ngpu
#        self.main = nn.Sequential(
#            # input is Z, going into a convolution
#            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
#            nn.BatchNorm2d(ngf * 16),
#            nn.ReLU(True),
#            # state size. (ngf*16) x 4 x 4
#            nn.ConvTranspose2d(ngf * 16 , ngf * 8, 4, 1, 0, bias=False),
#            nn.BatchNorm2d(ngf * 8),
#            nn.ReLU(True),
#            # state size. ``(ngf*8) x 8 x 8``
#            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 4),
#            nn.ReLU(True),
#            # state size. ``(ngf*4) x 16 x 16``
#            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 2),
#            nn.ReLU(True),
#            # state size. ``(ngf*2) x 32 x 32``
#            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),
#            # state size. ``(ngf) x 64 x 64``
#            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#            nn.Tanh()
#            # state size. ``(nc) x 128 x 128``
#        )
#    def forward(self, input):
#        return self.main(input)
#


# Discriminator

# class Discriminator(nn.Module):
#    def __init__(self, ngpu, nc, ndf):
#        super(Discriminator, self).__init__()
#        self.ngpu = ngpu
#        self.main = nn.Sequential(
#            # input is ``(nc) x 128 x 128``
#            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. ``(ndf) x 64 x 64``
#            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
#            nn.BatchNorm2d(ndf * 2),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. ``(ndf*2) x 32 x 32``
#            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
#            nn.BatchNorm2d(ndf * 4),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. ``(ndf*4) x 16 x 16``
#            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
#            nn.BatchNorm2d(ndf * 8),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. ``(ndf*8) x 8 x 8``
#            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
#            nn.BatchNorm2d(ndf * 16),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. ``(ndf*16) x 4 x 4``
#            nn.Conv2d(ndf * 16, 1, 4, stride=1,padding=0, bias=False),
#            nn.Sigmoid()
#        )
#    def forward(self, input):
#        return self.main(input)
