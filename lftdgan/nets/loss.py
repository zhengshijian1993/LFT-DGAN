import torch
import os
import math
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
import scipy.stats as st

def Weights_Normal(m):
    # initialize weights as Normal(mean, std)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


##############################################################################
#
#                              WGAN_GP
#
##############################################################################
class Gradient_Penalty(nn.Module):
    """ Calculates the gradient penalty loss for WGAN GP
    """
    def __init__(self, cuda=True):
        super(Gradient_Penalty, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, D, real, fake):
        # Random weight term for interpolation between real and fake samples
        eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True,)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


#####################################
#
#             PerceptualLoss
#
####################################
from torchvision.models.vgg import vgg19,vgg16
class VGG19_PercepLoss(nn.Module):
    def __init__(self):
        super(VGG19_PercepLoss,self).__init__()
        self.L1 = nn.L1Loss().cuda()
        self.mse = nn.MSELoss().cuda()
        vgg = vgg19(pretrained=True).eval().cuda()
        self.loss_net1 = nn.Sequential(*list(vgg.features)[:1]).eval().cuda()
        self.loss_net3 = nn.Sequential(*list(vgg.features)[:3]).eval().cuda()
        self.loss_net5 = nn.Sequential(*list(vgg.features)[:5]).eval().cuda()
        self.loss_net9 = nn.Sequential(*list(vgg.features)[:9]).eval().cuda()
        self.loss_net13 = nn.Sequential(*list(vgg.features)[:13]).eval().cuda()
    def forward(self,x,y):
        loss1 = self.L1(self.loss_net1(x),self.loss_net1(y))
        loss3 = self.L1(self.loss_net3(x),self.loss_net3(y))
        loss5 = self.L1(self.loss_net5(x),self.loss_net5(y))
        loss9 = self.L1(self.loss_net9(x),self.loss_net9(y))
        loss13 = self.L1(self.loss_net13(x),self.loss_net13(y))
        #print(self.loss_net13(x).shape)
        loss = 0.2*loss1 + 0.2*loss3 + 0.2*loss5 + 0.2*loss9 + 0.2*loss13
        return loss



###################################################################
#
#                     Gdl_loss
#
##################################################################

class Gradient_Difference_Loss(nn.Module):
    def __init__(self, alpha=1, chans=3, cuda=True):
        super(Gradient_Difference_Loss, self).__init__()
        self.alpha = alpha
        self.chans = chans
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
        self.Kx = Tensor(SobelX).expand(self.chans, 1, 3, 3)
        self.Ky = Tensor(SobelY).expand(self.chans, 1, 3, 3)

    def get_gradients(self, im):
        gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
        gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
        return gx, gy

    def forward(self, pred, true):
        # get graduent of pred and true
        gradX_true, gradY_true = self.get_gradients(true)
        grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
        gradX_pred, gradY_pred = self.get_gradients(pred)
        grad_pred_a = torch.abs(gradX_pred)**self.alpha + torch.abs(gradY_pred)**self.alpha
        # compute and return GDL
        return 0.5 * torch.mean(grad_true - grad_pred_a)


