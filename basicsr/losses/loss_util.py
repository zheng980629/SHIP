import functools
import torch
import numpy as np
from torch.nn import functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


def standardization(x):
    return (x - x.min()) / (x.max() - x.min())


def histcal(x, bins=256, min=0.0, max=1.0):
    n,c,h,w = x.size()
    n_batch = n
    row_m = h
    row_n = w
    channels = c

    delta = (max - min) / bins
    BIN_Table = np.arange(0, bins, 1)
    BIN_Table = BIN_Table * delta

    zero = torch.tensor([[[0.0]]],requires_grad=False).cuda()
    zero = zero.repeat(n,c,1)
    temp = torch.ones(size=x.size()).cuda()
    temp1 = torch.zeros(size=x.size()).cuda()
    for dim in range(1, bins - 1, 1):
        h_r = BIN_Table[dim]  # h_r
        h_r_sub_1 = BIN_Table[dim - 1]  # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1]  # h_(r+1)

        h_r = torch.tensor(h_r).float().cuda()
        h_r_sub_1 = torch.tensor(h_r_sub_1).float().cuda()
        h_r_plus_1 = torch.tensor(h_r_plus_1).float().cuda()

        h_r_temp = h_r * temp
        h_r_sub_1_temp = h_r_sub_1 * temp
        h_r_plus_1_temp = h_r_plus_1 * temp

        mask_sub = torch.where(torch.gt(h_r_temp, x) & torch.gt(x, h_r_sub_1_temp), temp, temp1)
        mask_plus = torch.where(torch.gt(x, h_r_temp) & torch.gt(h_r_plus_1_temp, x), temp, temp1)

        temp_mean1 = torch.mean((((x - h_r_sub_1) * mask_sub).view(n_batch, channels, -1)), dim=-1)
        temp_mean2 = torch.mean((((h_r_plus_1 - x) * mask_plus).view(n_batch, channels, -1)), dim=-1)

        if dim == 1:
            temp_mean = torch.add(temp_mean1, temp_mean2)
            temp_mean = torch.unsqueeze(temp_mean, -1)  # [1,1,1]
        else:
            if dim != bins - 2:
                temp_mean_temp = torch.add(temp_mean1, temp_mean2)
                temp_mean_temp = torch.unsqueeze(temp_mean_temp, -1)
                temp_mean = torch.cat([temp_mean, temp_mean_temp], dim=-1)
            else:
                zero = torch.cat([zero, temp_mean], dim=-1)
                temp_mean_temp = torch.add(temp_mean1, temp_mean2)
                temp_mean_temp = torch.unsqueeze(temp_mean_temp, -1)
                temp_mean = torch.cat([temp_mean, temp_mean_temp], dim=-1)

    # diff = torch.abs(temp_mean - zero)
    return temp_mean


def histcal_tensor(x, bins=256):
    N, C, H, W = x.shape
    x = x.view(N, -1)
    x_min, _ = x.min(-1) 
    x_min = x_min.unsqueeze(-1) 
    x_max, _ = x.max(-1)
    x_max = x_max.unsqueeze(-1) 
    q_levels = torch.arange(bins).float().cuda() 
    q_levels = q_levels.expand(N, bins)
    q_levels =  (2 * q_levels + 1) / (2 * bins) * (x_max - x_min) + x_min
    q_levels = q_levels.unsqueeze(1)
    q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] 
    q_levels_inter = q_levels_inter.unsqueeze(-1)
    x = x.unsqueeze(-1)
    quant = 1 - torch.abs(q_levels - x)
    quant = quant * (quant > (1 - q_levels_inter))
    sta = quant.sum(1) 
    sta = sta / (sta.sum(-1).unsqueeze(-1))

    return sta


def noiseMap(x):
    def sub_gradient(x):
        left_shift_x, right_shift_x, grad = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        left_shift_x[:, :, 0:-1] = x[:, :, 1:]
        right_shift_x[:, :, 1:] = x[:, :, 0:-1]
        grad = 0.5 * (left_shift_x - right_shift_x)
        return grad

    dx, dy = sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)
    noise_map = torch.max(dx.abs(), dy.abs())

    return noise_map


def rgb2lab(rgb):
    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :]
    b = rgb[:, 2, :, :]

    # gamma 2.2
    r = torch.where(r > 0.04045, torch.pow((r + 0.055) / 1.055, 2.4), r / 12.92)
    g = torch.where(g > 0.04045, torch.pow((g + 0.055) / 1.055, 2.4), g / 12.92)
    b = torch.where(b > 0.04045, torch.pow((b + 0.055) / 1.055, 2.4), b / 12.92)

    # sRGB
    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470

    # XYZ range: 0~100
    X = X * 100.000
    Y = Y * 100.000
    Z = Z * 100.000

    # Reference White Point

    ref_X = 96.4221
    ref_Y = 100.000
    ref_Z = 82.5211

    X = X / ref_X
    Y = Y / ref_Y
    Z = Z / ref_Z

    X = torch.where(X > 0.008856, torch.pow(X, 1 / 3.000), (7.787 * X) + (16 / 116.000))
    Y = torch.where(Y > 0.008856, torch.pow(Y, 1 / 3.000), (7.787 * Y) + (16 / 116.000))
    Z = torch.where(Z > 0.008856, torch.pow(Z, 1 / 3.000), (7.787 * Z) + (16 / 116.000))

    Lab_L = (116.000 * Y) - 16.000
    Lab_a = 500.000 * (X - Y)
    Lab_b = 200.000 * (Y - Z)
    
    return torch.cat((Lab_L.unsqueeze(dim=1), Lab_a.unsqueeze(dim=1), Lab_b.unsqueeze(dim=1)), dim=1)


def rgb2hsv(img, eps=1e-8):
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)

    return hsv


def rgb_to_grayscale(tensor):
    tensor = tensor.cpu()
    rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140])
    img_gray = torch.tensordot(tensor, rgb_weights, dims=([1], [0]))
    img_gray = torch.unsqueeze(img_gray, 1).cuda()
    return img_gray


def gradient(input_tensor, direction):
    smooth_kernel_x = torch.reshape(torch.Tensor([[0., 0.], [-1., 1.]]), (1, 1, 2, 2)).cuda()
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2).cuda()

    assert direction in ['x', 'y']
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y

    gradient_orig = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

def gaussian_kernel(kernel_size = 7, sig = 1.0):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    center = kernel_size//2, 
    x_axis = np.linspace(0, kernel_size-1, kernel_size) - center
    y_axis = np.linspace(0, kernel_size-1, kernel_size) - center
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig)) / (np.sqrt(2*np.pi)*sig)
    return kernel