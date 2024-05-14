import collections.abc
import math
import torch
import torchvision
import warnings
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Noise2Noise_ConvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride=1, pad=1, use_act=True):
        super(Noise2Noise_ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(ni, no, ks, stride=stride, padding=pad)
        self.act = nn.ReLU()

    def forward(self, x):
        op = self.conv(x)
        return self.act(op) if self.use_act else op


def crop_and_concat(upsampled, bypass, crop=True):
    if crop:
        if (bypass.size()[3] - upsampled.size()[3]) / 2 == 0.5:
            c = 1
        else:
            c = (bypass.size()[3] - upsampled.size()[3]) // 2
        if (bypass.size()[2] - upsampled.size()[2]) / 2 == 0.5:
            cc = 1
        else:
            cc = (bypass.size()[2] - upsampled.size()[2]) // 2
        assert (c >= 0)
        assert (cc >= 0)
        if c == 1 or cc == 1:
            upsampled = F.pad(upsampled, (c, 0, cc, 0))
        else:
            upsampled = F.pad(upsampled, (c, c, cc, cc))
    return torch.cat((upsampled, bypass), 1)

class Conv3x3Stack(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(Conv3x3Stack, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, inputs):
        return self.block(inputs)


class DConv3x3Stack(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(DConv3x3Stack, self).__init__()
        self.DConv = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), stride=(2, 2), padding=0, dilation=1, groups=1, bias=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.LeakyReLU(negative_slope)
        )

    def forward(self, x1, x2):
        upsample_x1 = self.DConv(x1)
        up = crop_and_concat(upsample_x1, x2)
        # up = torch.cat((upsample_x1, x2), 1)
        return self.block(up)


class UpConv3x3Stack(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(UpConv3x3Stack, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.LeakyReLU(negative_slope)
        )

    def forward(self, x1, x2):
        upsample_x1 = self.DConv(x1)
        up = crop_and_concat(upsample_x1, x2)
        # up = torch.cat((upsample_x1, x2), 1)
        return self.block(up)


class Half_Exposure_Interactive_Modulation(nn.Module):
    def __init__(self, vector_dim, feature_channel):
        super(Half_Exposure_Interactive_Modulation, self).__init__()
        self.vector_dim = vector_dim
        self.feature_channel = feature_channel

        self.modulation_mul = nn.Sequential(
            nn.Linear(vector_dim, feature_channel // 2)
        )

    def forward(self, feature, modulation_vector):
        feature_modulation = feature[:, :self.feature_channel // 2, :, :]
        feature_identity = feature[:,self.feature_channel // 2:, :, :]

        modulation_vector_mul = self.modulation_mul(modulation_vector)
        feature_attention = torch.sigmoid(torch.mul(feature_modulation, modulation_vector_mul.unsqueeze(2).unsqueeze(3)))

        return torch.cat((feature_attention * feature_modulation, feature_identity), dim=1)


class Denoise_Interactive_Modulation(nn.Module):
    def __init__(self, vector_dim, feature_channel):
        super(Denoise_Interactive_Modulation, self).__init__()
        self.vector_dim = vector_dim
        self.feature_channel = feature_channel

        self.modulation_mul = nn.Sequential(
            nn.Linear(vector_dim, feature_channel, bias=False)
        )

    def forward(self, feature, modulation_vector):

        modulation_vector_mul = self.modulation_mul(modulation_vector)
        feature_attention = torch.sigmoid(torch.mul(feature, modulation_vector_mul.unsqueeze(2).unsqueeze(3)))

        return feature_attention * feature


class Interactive_Modulation(nn.Module):
    def __init__(self, vector_dim, feature_channel):
        super(Interactive_Modulation, self).__init__()
        self.vector_dim = vector_dim
        self.feature_channel = feature_channel
        self.conv = nn.Conv2d(feature_channel, feature_channel, 3, 1, 1)

        self.modulation_mul = nn.Sequential(
            nn.Linear(vector_dim, feature_channel, bias=False)
        )

    def forward(self, feature, modulation_vector):

        modulation_vector_mul = self.modulation_mul(modulation_vector)

        feature_attention = torch.sigmoid(torch.mul(self.conv(feature), modulation_vector_mul.unsqueeze(2).unsqueeze(3)))

        return feature_attention


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvLReLUNoBN(nn.Module):
    """
    It has a style of:
        ---Conv-LeakyReLU---

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, input_channel, output_channel, kernel=3, stride=1, padding=1, pytorch_init=False, act=True, negative_slope=0.2):
        super(ConvLReLUNoBN, self).__init__()
        self.act = act
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel, stride, padding, bias=True)
        if self.act:
            self.relu = nn.LeakyReLU(negative_slope)

        if not pytorch_init:
            default_init_weights([self.conv1], 0.1)

    def forward(self, x):
        if self.act:
            out = self.relu(self.conv1(x))
        else: 
            out = self.conv1(x)
        return out


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class upsample_and_concat(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(upsample_and_concat, self).__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self. deconv = nn.ConvTranspose2d(self.in_channels, self.output_channels, (2, 2), stride=(2, 2), padding=0, dilation=1, groups=1, bias=True)
        self.upsample = F.upsample_nearest
    
    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        x1 = self.upsample(x1, x2.shape[-2:])
        return torch.cat((x1, x2), dim=1)


class Half_Illumination_Interactive_Modulation(nn.Module):
    def __init__(self, feature_channel, dims):
        super(Half_Illumination_Interactive_Modulation, self).__init__()
        self.feature_channel = feature_channel

        self.conditional_fc = nn.Sequential(
            nn.Linear(dims, self.feature_channel // 2, bias=False), 
            nn.Linear(self.feature_channel // 2, self.feature_channel // 2, bias=False)
        )

    def forward(self, feature, histogram_vector):
        modulation_vector = self.conditional_fc(histogram_vector)

        feature_modulation = feature[:, :self.feature_channel // 2, :, :]
        feature_identity = feature[:,self.feature_channel // 2:, :, :]

        feature_attention = torch.sigmoid(torch.mul(feature_modulation, modulation_vector.unsqueeze(2).unsqueeze(3)))

        return torch.cat((feature_attention * feature_modulation, feature_identity), dim=1)

def simple_batch_norm_1d(x):
    eps = 1e-10
    x_mean = torch.mean(x, dim=1, keepdim=True) # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=1, keepdim=True)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    return x_hat


class PONO(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=True, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std


class PONO_woNorm(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=True, eps=1e-5):
        super(PONO_woNorm, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        # x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std

class MS(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MS, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if torchvision.__version__ >= '0.9.0':
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
