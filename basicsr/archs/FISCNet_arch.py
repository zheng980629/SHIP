from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F

from collections import OrderedDict

from math import exp
from .utils.CDC import cdcconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine, CALayer
import torch.nn.init as init
import os
import cv2

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class DenseBlockMscale(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier'):
        super(DenseBlockMscale, self).__init__()
        self.ops = DenseBlock(channel_in, channel_out, init)
        self.fusepool = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc1 = nn.Sequential(nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fc3 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fuse = nn.Conv2d(3*channel_out,channel_out,1,1,0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')
        x1 = self.ops(x1)
        x2 = self.ops(x2)
        x3 = self.ops(x3)
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        xattw = self.fusepool(x1+x2+x3)
        xattw1 = self.fc1(xattw)
        xattw2 = self.fc2(xattw)
        xattw3 = self.fc3(xattw)
        # x = x1*xattw1+x2*xattw2+x3*xattw3
        x = self.fuse(torch.cat([x1*xattw1,x2*xattw2,x3*xattw3],1))

        return x


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlockMscale(channel_in, channel_out, init)
            else:
                return DenseBlockMscale(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None
    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class Interaction(nn.Module):
    def __init__(self, channels):
        super(Interaction, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels

        self.spa_att_vis = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, 3, 1, 1, bias=True),
                                     nn.Sigmoid())
        self.cha_att_vis = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, 1),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, 1),
                                     nn.Sigmoid())
        self.post_vis = nn.Conv2d(channels * 2, channels, 3, 1, 1)

        self.spa_att_inf = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, 1, 1),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, 3, 1, 1),
                                     nn.Sigmoid())
        
        self.cha_att_inf = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, 1),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, 1),
                                     nn.Sigmoid())
        self.post_inf = nn.Conv2d(channels * 2, channels, 3, 1, 1)

        self.fused = InvBlock(subnet('DBNet'), channels * 2, channels)
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, fused_y, vis_y, inf):
        vis_map = self.spa_att_vis(vis_y - fused_y)
        vis_res = vis_map * fused_y + fused_y
        vis_cat = torch.cat([vis_res, fused_y], 1)
        vis_cha =  self.post_vis(self.cha_att_vis(self.contrast(vis_cat) + self.avgpool(vis_cat)) * vis_cat)
        vis_out = vis_cha + fused_y

        inf_map = self.spa_att_inf(inf - fused_y)
        inf_res = inf_map * fused_y + fused_y
        inf_cat = torch.cat([inf_res, fused_y], 1)
        inf_cha =  self.post_inf(self.cha_att_inf(self.contrast(inf_cat) + self.avgpool(inf_cat)) * inf_cat)
        inf_out = inf_cha + fused_y

        cat = torch.cat([vis_out, inf_out], 1)
        fused = self.post(self.fused(cat))

        return fused


class EdgeBlock(nn.Module):
    def __init__(self, channelin, channelout):
        super(EdgeBlock, self).__init__()
        self.process = nn.Conv2d(channelin,channelout,3,1,1)
        self.Res = nn.Sequential(nn.Conv2d(channelout,channelout,3,1,1),
            nn.ReLU(),nn.Conv2d(channelout, channelout, 3, 1, 1))
        self.CDC = cdcconv(channelout, channelout)

    def forward(self,x):

        x = self.process(x)
        out = self.Res(x) + self.CDC(x)

        return out


class Calibration_first(nn.Module):
    def __init__(self, channelin, channelout):
        super(Calibration_first, self).__init__()
        self.vis = EdgeBlock(channelin,channelout)
        self.inf = EdgeBlock(channelin, channelout)
        self.fused = EdgeBlock(channelin, channelout)
        self.interaction = Interaction(channelout)

    def forward(self, fused_y, vis_y, inf, i, j):

        fused_y = self.fused(fused_y)
        vis_y = self.fused(vis_y)
        inf = self.fused(inf)
        fused_calbrated = self.interaction(fused_y, vis_y, inf)
        
        return fused_calbrated, vis_y, inf
    

class Calibration(nn.Module):
    def __init__(self, channelin, channelout):
        super(Calibration, self).__init__()
        self.vis = EdgeBlock(channelin, channelout)
        self.inf = EdgeBlock(channelin, channelout)
        self.fused = EdgeBlock(channelin, channelout)
        self.interaction = Interaction(channelin)

    def forward(self, fused_y, vis_y, inf, i, j):

        fused_y = self.fused(fused_y)
        vis_y = self.vis(vis_y)
        inf = self.inf(inf)
        
        fused_fft = torch.fft.fft2(fused_y)
        fused_abs = torch.abs(fused_fft)
        fused_pha = torch.angle(fused_fft)

        vis_y_fft = torch.fft.fft2(vis_y)
        vis_abs = torch.abs(vis_y_fft)
        vis_pha = torch.angle(vis_y_fft)

        inf_fft = torch.fft.fft2(inf)
        inf_abs = torch.abs(inf_fft)
        inf_pha = torch.angle(inf_fft)

        fused_a = fused_abs * (torch.cos(vis_pha) + torch.cos(inf_pha))
        fused_b = fused_abs * (torch.sin(vis_pha) + torch.sin(inf_pha))

        fused_y = torch.complex(fused_a, fused_b)
        fused_y = torch.fft.ifft2(fused_y)
        fused_y = torch.abs(fused_y)

        fused_calbrated = self.interaction(fused_y, vis_y, inf)

        
        return fused_calbrated, vis_y, inf
    



@ARCH_REGISTRY.register()
class FISCNet(nn.Module):
    def __init__(self, vis_channels=1, inf_channels=1,
                 n_feat=16):
        super(FISCNet,self).__init__()

        self.correction1 = Calibration_first(channelin=vis_channels, channelout=n_feat)
        self.correction2 = Calibration(channelin=n_feat, channelout=n_feat)
        self.correction3 = Calibration(channelin=n_feat, channelout=n_feat)
        self.reconstruction = Refine(n_feat, out_channels=vis_channels)
        self.i = 0

    def forward(self, image_vis, image_ir):

        vis_y = image_vis[:,:1]
        inf = image_ir

        vis_y_fft = torch.fft.fft2(vis_y)
        vis_abs = torch.abs(vis_y_fft)
        vis_pha = torch.angle(vis_y_fft)

        inf_fft = torch.fft.fft2(inf)
        inf_abs = torch.abs(inf_fft)
        inf_pha = torch.angle(inf_fft)

        fused_a = vis_abs * (torch.cos(vis_pha) + torch.cos(inf_pha))
        fused_b = vis_abs * (torch.sin(vis_pha) + torch.sin(inf_pha))

        fused_y = torch.complex(fused_a, fused_b)
        fused_y = torch.fft.ifft2(fused_y)
        fused_y = torch.abs(fused_y)

        fused_y1, vis_y_feat, inf_feat = self.correction1(fused_y, vis_y, inf, self.i, j=1)
        fused_y2, vis_y_feat2, inf_feat2 = self.correction2(fused_y1, vis_y_feat, inf_feat, self.i, j=2)
        fused_y3, vis_y_feat3, inf_feat3 = self.correction3(fused_y2, vis_y_feat2, inf_feat2, self.i, j=3)

        fused = self.reconstruction(fused_y3)

        self.i += 1

        return fused

