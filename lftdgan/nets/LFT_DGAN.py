
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_dct

from torch.nn.utils import spectral_norm
from .LFT_generator import LFT_generator

##############################################################################
#
#                             Discriminator Code
#
##############################################################################

class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


##############################################################################
#
#                              Spectral Discriminator Code
#
##############################################################################

import math
from torch.autograd import Variable

def shift(x: torch.Tensor):
    out = torch.zeros_like(x)

    H, W = x.size(-2), x.size(-1)
    out[:,:int(H/2),:int(W/2)] = x[:,int(H/2):,int(W/2):]
    out[:,:int(H/2),int(W/2):] = x[:,int(H/2):,:int(W/2)]
    out[:,int(H/2):,:int(W/2)] = x[:,:int(H/2),int(W/2):]
    out[:,int(H/2):,int(W/2):] = x[:,:int(H/2),:int(W/2)]
    return out

def RGB2gray(rgb):
    if rgb.size(1) == 3:
        r, g, b = rgb[:,0,:,:], rgb[:,1,:,:], rgb[:,2,:,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    elif rgb.size(1) == 1:
        return rgb[:,0,:,:]

# Azimutal Averaging Operation
def azimuthalAverage(image, center=None):
    # Calculate the indices from the image
    H, W = image.shape[0], image.shape[1]
    y, x = np.indices([H, W])
    radius = np.sqrt((x - H/2)**2 + (y - W/2)**2)
    radius = radius.astype(np.int).ravel()
    nr = np.bincount(radius)
    tbin = np.bincount(radius, image.ravel())
    radial_prof = tbin / (nr + 1e-10)
    return radial_prof[1:-2]

def get_fft_feature(x):
    x_rgb = x.detach()
    epsilon = 1e-8

    x_gray = RGB2gray(x_rgb)
    # fft = torch.rfft(x_gray, 2, onesided=False)
    output_fft_new = torch.fft.fft2(x_gray, dim=(-2, -1))
    fft = torch.stack((output_fft_new.real, output_fft_new.imag), -1)
    fft += epsilon
    magnitude_spectrum = torch.log((torch.sqrt(fft[:, :, :, 0] ** 2 + fft[:, :, :, 1] ** 2 + 1e-10)) + 1e-10)
    magnitude_spectrum = shift(magnitude_spectrum)
    magnitude_spectrum = magnitude_spectrum.cpu().numpy()

    out = []
    for i in range(magnitude_spectrum.shape[0]):
        out.append(torch.from_numpy(azimuthalAverage(magnitude_spectrum[i])).float().unsqueeze(0))
    out = torch.cat(out, dim=0)

    out = (out - torch.min(out, dim=1, keepdim=True)[0]) / (
                torch.max(out, dim=1, keepdim=True)[0] - torch.min(out, dim=1, keepdim=True)[0])
    out = Variable(out, requires_grad=True).to(x.device)

    return out



class Spectral_Discriminator(nn.Module):
    def __init__(self, height):
        super(Spectral_Discriminator, self).__init__()
        self.thresh = int(height / (2 * math.sqrt(2)))
        self.linear = nn.Linear(self.thresh, 1)


    def forward(self, input: torch.Tensor):

        input = torch_dct.dct_2d(input)                 #############3
        az_fft_feature = get_fft_feature(input)
        az_fft_feature[torch.isnan(az_fft_feature)] = 0

        return self.linear(az_fft_feature[:, -self.thresh:])




class LFT_DGAN:
    def __init__(self, base_model='lft-dgan'):
        if base_model=='lft-dgan': # default
            self.netG = LFT_generator()

            self.netD = UNetDiscriminatorSN(3)
            self.netD_s = Spectral_Discriminator(256)
        elif base_model=='resnet':
            #TODO: add ResNet support
            pass
        else:
            pass