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
import numbers
from einops import rearrange
import numpy

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
    
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class spatialInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(spatialInteraction, self).__init__()
        self.reflashFused1 = nn.Sequential(
                            nn.Conv2d(channelin, channelout, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout, channelout, 3, 1, 1)
                        )
        self.reflashFused2 = nn.Sequential(
                            nn.Conv2d(channelin, channelout, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout, channelout, 3, 1, 1)
                        )
        self.reflashFused3 = nn.Sequential(
                            nn.Conv2d(channelin, channelout, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout, channelout, 3, 1, 1)
                        )
        self.reflashInfrared1 = nn.Sequential(
                            nn.Conv2d(channelin, channelout, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout, channelout, 3, 1, 1)
                        )
        self.reflashInfrared2 = nn.Sequential(
                            nn.Conv2d(channelin, channelout, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout, channelout, 3, 1, 1)
                        )
        self.reflashInfrared3 = nn.Sequential(
                            nn.Conv2d(channelin, channelout, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout, channelout, 3, 1, 1)
                        )
        
        self.norm1 = LayerNorm(channelout, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(channelout, LayerNorm_type='WithBias')
        self.norm3 = LayerNorm(channelout, LayerNorm_type='WithBias')
        self.norm4 = LayerNorm(channelout, LayerNorm_type='WithBias')
        

    def forward(self, vis, inf, i, j):

        _, C, H, W = vis.size()

        vis_fft = torch.fft.rfft2(vis.float())
        inf_fft = torch.fft.rfft2(inf.float())

        atten = vis_fft * inf_fft
        atten = torch.fft.irfft2(atten, s=(H, W))
        atten = self.norm1(atten)
        fused_OneOrderSpa = atten * inf

        fused_OneOrderSpa = self.reflashFused1(fused_OneOrderSpa)
        fused_OneOrderSpa = self.norm2(fused_OneOrderSpa)
        infraredReflash1 = self.reflashInfrared1(inf)
        fused_twoOrderSpa = fused_OneOrderSpa * infraredReflash1

        fused_twoOrderSpa = self.reflashFused2(fused_twoOrderSpa)
        fused_twoOrderSpa = self.norm3(fused_twoOrderSpa)
        infraredReflash2 = self.reflashInfrared2(infraredReflash1)
        fused_threeOrderSpa = fused_twoOrderSpa * infraredReflash2

        fused_threeOrderSpa = self.reflashFused3(fused_threeOrderSpa)
        fused_threeOrderSpa = self.norm4(fused_threeOrderSpa)
        infraredReflash3 = self.reflashInfrared3(infraredReflash2)
        fused_fourOrderSpa = fused_threeOrderSpa * infraredReflash3

        fused = fused_fourOrderSpa + vis

        return fused, infraredReflash3
    

class channelInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(channelInteraction, self).__init__()
        self.chaAtten = nn.Sequential(nn.Conv2d(channelin * 2, channelout, kernel_size=1, padding=0, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(channelout, channelin * 2, kernel_size=1, padding=0, bias=True))
        self.reflashChaAtten1 = nn.Sequential(nn.Conv2d(channelin * 2, channelout, kernel_size=1, padding=0, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(channelout, channelin * 2, kernel_size=1, padding=0, bias=True))
        self.reflashChaAtten2 = nn.Sequential(nn.Conv2d(channelin * 2, channelout, kernel_size=1, padding=0, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(channelout, channelin * 2, kernel_size=1, padding=0, bias=True))
        self.reflashChaAtten3 = nn.Sequential(nn.Conv2d(channelin * 2, channelout, kernel_size=1, padding=0, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(channelout, channelin * 2, kernel_size=1, padding=0, bias=True))
        
        self.reflashFused1 = nn.Sequential(
                            nn.Conv2d(channelin * 2, channelout * 2, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout * 2, channelout * 2, 3, 1, 1)
                        )
        self.reflashFused2 = nn.Sequential(
                            nn.Conv2d(channelin * 2, channelout * 2, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout * 2, channelout * 2, 3, 1, 1)
                        )
        self.reflashFused3 = nn.Sequential(
                            nn.Conv2d(channelin * 2, channelout * 2, 3, 1, 1),
                            nn.ReLU(),
                            nn.Conv2d(channelout * 2, channelout * 2, 3, 1, 1)
                        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.postprocess = nn.Sequential(InvBlock(DenseBlock, 2 * channelin, channelout),
                                         nn.Conv2d(2*channelout, channelout, 1, 1, 0))
        
    def forward(self, vis, inf, i, j):

        vis_cat = torch.cat([vis, inf], 1)

        chanAtten = self.chaAtten(self.avgpool(vis_cat)).softmax(1)
        channel_response = self.chaAtten(self.avgpool(vis_cat))
        fused_OneOrderCha = vis_cat * chanAtten

        fused_OneOrderCha = self.reflashFused1(fused_OneOrderCha)
        chanAttenReflash1 = self.reflashChaAtten1(chanAtten).softmax(1)
        fused_twoOrderCha = fused_OneOrderCha * chanAttenReflash1

        fused_twoOrderCha = self.reflashFused2(fused_twoOrderCha)
        chanAttenReflash2 = self.reflashChaAtten2(chanAttenReflash1).softmax(1)
        fused_threeOrderCha = fused_twoOrderCha * chanAttenReflash2

        fused_threeOrderCha = self.reflashFused3(fused_threeOrderCha)
        chanAttenReflash3 = self.reflashChaAtten3(chanAttenReflash2).softmax(1)
        fused_fourOrderCha = fused_threeOrderCha * chanAttenReflash3

        fused_fourOrderCha = self.postprocess(fused_fourOrderCha)

        fused = fused_fourOrderCha + vis

        return fused, inf
    

class highOrderInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(highOrderInteraction, self).__init__()
        self.spatial = spatialInteraction(channelin, channelout)
        self.channel = channelInteraction(channelin, channelout)

    def forward(self, vis_y, inf, i, j):

        vis_spa, inf_spa = self.spatial(vis_y, inf, i, j)
        vis_cha, inf_cha = self.channel(vis_spa, inf_spa, i, j)
        
        return vis_cha, inf_cha
    

class EdgeBlock(nn.Module):
    def __init__(self, channelin, channelout):
        super(EdgeBlock, self).__init__()
        self.process = nn.Conv2d(channelin,channelout,3,1,1)
        self.Res = nn.Sequential(nn.Conv2d(channelout,channelout,3,1,1),
            nn.ReLU(),nn.Conv2d(channelout, channelout, 3, 1, 1))
        self.CDC = cdcconv(channelin, channelout)

    def forward(self,x):

        x = self.process(x)
        out = self.Res(x) + self.CDC(x)

        return out


class FeatureExtract(nn.Module):
    def __init__(self, channelin, channelout):
        super(FeatureExtract, self).__init__()
        self.conv = nn.Conv2d(channelin,channelout,1,1,0)
        self.block1 = EdgeBlock(channelout,channelout)
        self.block2 = EdgeBlock(channelout, channelout)

    def forward(self,x):
        xf = self.conv(x)
        xf1 = self.block1(xf)
        xf2 = self.block2(xf1)

        return xf2


@ARCH_REGISTRY.register()
class highOrderInteractionFusion(nn.Module):
    def __init__(self, vis_channels=1, inf_channels=1,
                 n_feat=16):
        super(highOrderInteractionFusion,self).__init__()

        
        self.vis = FeatureExtract(vis_channels, n_feat)
        self.inf = FeatureExtract(inf_channels, n_feat)

        self.interaction1 = highOrderInteraction(channelin=n_feat, channelout=n_feat)
        self.interaction2 = highOrderInteraction(channelin=n_feat, channelout=n_feat)
        self.interaction3 = highOrderInteraction(channelin=n_feat, channelout=n_feat)

        self.postprocess = nn.Sequential(InvBlock(DenseBlock, 3 * n_feat, 3 * n_feat // 2),
                                         nn.Conv2d(3 * n_feat, n_feat, 1, 1, 0))
        
        self.reconstruction = Refine(n_feat, out_channels=vis_channels)

        self.i = 0

    def forward(self, image_vis, image_ir):

        vis_y = image_vis[:,:1]
        inf = image_ir

        vis_y = self.vis(vis_y)
        inf = self.inf(inf)

        vis_y_feat, inf_feat = self.interaction1(vis_y, inf, self.i, j=1)
        vis_y_feat2, inf_feat2 = self.interaction2(vis_y_feat, inf_feat, self.i, j=2)
        vis_y_feat3, inf_feat3 = self.interaction3(vis_y_feat2, inf_feat2, self.i, j=3)

        fused = self.postprocess(torch.cat([vis_y_feat, vis_y_feat2, vis_y_feat3], 1))

        fused = self.reconstruction(fused)

        self.i += 1

        return fused

