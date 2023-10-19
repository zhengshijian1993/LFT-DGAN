import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv1x1(out_channels, out_channels//2)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        out = self.conv3(out)
        return out


class conv2dwithActivation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv2dwithActivation, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)

        return out


#########################################################################

# similarity
class norm_attention(nn.Module):
    def __init__(self, dim, kernel_size, stride):
        super(norm_attention, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dim = dim
        reduction_ratio = 2
        self.groups_channels = 8
        self.groups = self.dim // self.groups_channels
        self.conv1 = nn.Conv2d(dim,dim//reduction_ratio,1)
        self.conv2 = nn.Conv2d(dim//reduction_ratio,kernel_size**2*self.groups, 1, 1)
        if stride >1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, input_x, input_y):
        # similarly
        weight = self.conv2(self.conv1(input_x if self.stride == 1 else self.avgpool(input_x)))    # channel up
        b, c, h, w = weight.shape
        weight = weight.view(b,self.groups,self.kernel_size**2,h,w).unsqueeze(2)
        out = self.unfold(input_y).view(b,self.groups,self.groups_channels,self.kernel_size**2,h,w)
        out = (weight*out).sum(dim=3).view(b,self.dim,h,w)
        return out

class LFT(nn.Module):
    def __init__(self,norm_nc, input_nc, ch_rate=4, size_rate=1):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        self.norm = nn.InstanceNorm2d(norm_nc//2, affine=False, track_running_stats=False)
        ks = 3
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = norm_nc // ch_rate

        pw = ks // 2
        self.mlp_shared0 = nn.Sequential(
            nn.Conv2d(norm_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_shared1 = nn.Sequential(
            nn.Conv2d(input_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.an = norm_attention(nhidden, kernel_size=ks, stride = 1)          # ch


    def forward(self, x, segmap):

        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map

        # gamma and beta
        actv0 = self.mlp_shared0(x)
        actv1 = self.mlp_shared1(segmap)

        actv = self.an(actv0, actv1)        # similarity

        out1,out2=torch.chunk(actv,2,dim=1)                                  # HINet
        actv = torch.cat(([self.norm(out1),out2]),dim=1)


        # actv = self.param_free_norm(actv)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, norm_layer=None, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.activation = activation
        if norm_layer is not None:
            self.conv2d = norm_layer(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
            self.mask_conv2d = norm_layer(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        else:
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
            self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                               groups, bias)
        # self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        return x


class LFT_ResnetBlock(nn.Module):
    def __init__(self, fin, fout, ch_rate=4, size_rate=1):
        super().__init__()


        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = GatedConv2dWithActivation(fin, fmiddle, kernel_size=3, padding=1, activation=None)
        self.conv_1 = GatedConv2dWithActivation(fmiddle, fout, kernel_size=3, padding=1, activation=None)

        # define normalization layers

        self.norm_0 = LFT(fin, fout, ch_rate, size_rate)
        self.norm_1 = LFT(fmiddle, fout, ch_rate, size_rate)


    def forward(self, x, seg):
        x_s = x

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        # dx = self.conv_0(self.actvn(self.norm_0(dx, seg)))
        # dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        #
        # dx = self.conv_0(self.actvn(self.norm_0(dx, seg)))
        # dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out


    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
