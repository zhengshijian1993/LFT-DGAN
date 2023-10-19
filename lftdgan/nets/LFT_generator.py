import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from .LFT_block import LFT_ResnetBlock

#############################################################################################3
#############################    DCT_split(low-high-mid)

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return in_a, out_b, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)

class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        z1, z2, _ = self.coupling(out)

        return z1, z2

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

def PONO(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output, mean, std
def MS(x, beta, gamma):
    return x * (gamma) + beta


class FD_norm(nn.Module):
    def __init__(self, in_channel, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 16

        self.norm = nn.InstanceNorm2d(squeeze_dim, affine=False, track_running_stats=False)


        self.flows = Flow(squeeze_dim)
        self.flows1 = Flow(squeeze_dim//2)
        self.flows2 = Flow(squeeze_dim // 4)


    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 4, 4, width // 4, 4)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 16, height // 4, width // 4)

        z1, z2= self.flows(out)
        z3, z4 = self.flows1(z2)
        z5, z6 = self.flows2(z4)
        z5 = torch.cat((z5, z6),dim=1)


        output, mean, std = PONO(z1)
        low = MS(out, mean, std)

        output, mean1, std1 = PONO(z3)
        mid = MS(out, mean1, std1)

        output, mean2, std2 = PONO(z5)
        hig = MS(out, mean2, std2)


        b_size1, n_channel1, height1, width1 = low.shape

        low = low.view(b_size1, n_channel1 // 16, 4, 4, height1, width1)
        low = low.permute(0, 1, 4, 2, 5, 3)
        low = low.contiguous().view(b_size1, n_channel1 // 16, height1 * 4, width1 * 4)

        mid = mid.view(b_size1, n_channel1 // 16, 4, 4, height1, width1)
        mid = mid.permute(0, 1, 4, 2, 5, 3)
        mid = mid.contiguous().view(b_size1, n_channel1 // 16, height1 * 4, width1 * 4)

        hig = hig.view(b_size1, n_channel1 // 16, 4, 4, height1, width1)
        hig = hig.permute(0, 1, 4, 2, 5, 3)
        hig = hig.contiguous().view(b_size1, n_channel1 // 16, height1 * 4, width1 * 4)

        out = torch.cat((low, mid, hig),dim=1)

        return out

####################################################### frequency_enhance_generator #######################3

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResnetBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(False), dilation=1, kernel_size=3):
        super().__init__()

        self.conv_block0 = nn.Sequential(nn.Conv2d(dim, dim,kernel_size=kernel_size, dilation=3, padding=3),activation)
        self.conv_block1 = nn.Sequential(nn.Conv2d(dim, dim,kernel_size=kernel_size, dilation=2, padding=2), activation)
        self.conv_block2 = nn.Sequential(nn.Conv2d(dim, dim,kernel_size=kernel_size, dilation=1, padding=1), activation)
        self.conv_block3 = nn.Conv2d(3*dim,dim,kernel_size=1)

        self.conv_block = nn.Sequential(conv3x3(dim, dim),nn.ReLU(inplace=True),conv1x1(dim, dim))


    def forward(self, x):

        x1 = self.conv_block0(x)
        x2 = self.conv_block1(x)
        x3 = self.conv_block2(x)
        out = torch.cat((x1,x2,x3),dim=1)
        y = self.conv_block3(out)
        out = x + y

        return out

class de_net(nn.Module):

    def __init__(self, nc):
        super(de_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(nc, nc, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(nc*2, nc, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(nc*2, nc, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(nc*4, nc, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(nc, 3, 1, 1, 0, bias=True)
    def forward(self, x):

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))

        clean_image = self.e_conv6(x5).tanh()
        return clean_image


class LFT_generator(nn.Module):
    def __init__(self):
        super(LFT_generator, self).__init__()

        # split_frequency
        self.FD_norm = FD_norm(3)

        self.conv = conv1x1(3,32)
        self.frequency_en = LFT_ResnetBlock(32, 32)
        self.GC = ResnetBlock(32)
        self.fu = de_net(32 * 3)

    def forward(self, image):

        # image divide 3 part

        fad = self.FD_norm(image)

        dct_LOW = self.conv(fad[:,0:3,:,:])
        dct_MID = self.conv(fad[:,3:6,:,:])
        dct_HIGH =self.conv(fad[:,6:9,:,:])

        H1 = self.GC(dct_LOW)
        H2 = self.frequency_en(dct_MID, H1)
        H3 = self.frequency_en(dct_HIGH, H2)


        out = torch.cat((H1, H2, H3),dim=1)
        out = self.fu(out)

        return out





