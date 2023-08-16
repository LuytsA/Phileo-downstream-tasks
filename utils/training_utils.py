import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import gc
gc.collect(2)
import numpy as np
import buteo as beo
import os
import sys; sys.path.append("..")
import config_lc

def patience_calculator(epoch, t_0, t_m, max_patience=50):
    """ Calculate the patience for the scheduler. """
    if epoch <= t_0:
        return t_0

    p = [t_0 * t_m ** i for i in range(100) if t_0 * t_m ** i <= epoch][-1]
    if p > max_patience:
        return max_patience

    return p


class TiledMSE(nn.Module):
    """
    Calculates the MSE at full image level and at the pixel level and weights the two.
    result = (sum_mse * (1 - bias)) + (mse * bias)
    """

    def __init__(self, bias=0.2):
        super(TiledMSE, self).__init__()
        self.bias = bias

    def forward(self, y_pred, y_true):
        y_pred_sum = torch.sum(y_pred, dim=(1, 2, 3)) / (y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3])
        y_true_sum = torch.sum(y_true, dim=(1, 2, 3)) / (y_true.shape[1] * y_true.shape[2] * y_true.shape[3])

        sum_mse = ((y_pred_sum - y_true_sum) ** 2).mean()
        mse = ((y_pred - y_true) ** 2).mean()

        weighted = (sum_mse * (1 - self.bias)) + (mse * self.bias)

        return weighted


def render_s2_as_rgb(arr, channel_first=False):
    # If there are nodata values, lets cast them to zero.
    if np.ma.isMaskedArray(arr):
        arr = np.ma.getdata(arr.filled(0))

    if channel_first:
        arr = beo.channel_first_to_last(arr)
    # Select only Blue, green, and red. Then invert the order to have R-G-B
    rgb_slice = arr[:, :, 0:3][:, :, ::-1]

    # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
    # Which produces dark images.
    rgb_slice = np.clip(
        rgb_slice,
        np.quantile(rgb_slice, 0.02),
        np.quantile(rgb_slice, 0.98),
    )

    # The current slice is uint16, but we want an uint8 RGB render.
    # We normalise the layer by dividing with the maximum value in the image.
    # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
    rgb_slice = (rgb_slice / rgb_slice.max()) * 255.0

    # We then round to the nearest integer and cast it to uint8.
    rgb_slice = np.rint(rgb_slice).astype(np.uint8)

    return rgb_slice


def visualise(x, y, y_pred=None, images=5, channel_first=False, vmin=0, vmax=1, save_path=None, for_landcover=False):
    rows = images
    cmap = 'magma'
    # print(x[0].shape,y[0].shape, y_pred[0].shape)
    if y_pred is None:
        columns = 2
    else:
        columns = 3
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))


    if for_landcover:
        lc_map_names = config_lc.lc_raw_classes
        lc_map = config_lc.lc_model_map
        lc_map_inverted = {v:k for k,v in zip(lc_map.keys(),lc_map.values())}
        vmax=len(lc_map)

        # d = 1 if channel_first else -1
        # # y= y.argmax(axis=d)
        # if y_pred is not None:
        #     y_pred = y_pred.argmax(axis=d)
        cmap = (matplotlib.colors.ListedColormap(config_lc.lc_color_map.values()))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


    for idx in range(0, images):
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(y[idx].squeeze(), vmin=vmin, vmax=vmax, cmap=cmap)
        if for_landcover:
            patches = [mpatches.Patch(color=cmap(norm(u)), label=lc_map_names[lc_map_inverted[u]]) for u in np.unique(y[idx])]
            plt.legend(handles=patches)
        plt.axis('on')
        plt.grid()

        if y_pred is not None:
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(y_pred[idx].squeeze(), vmin=vmin, vmax=vmax, cmap=cmap)
            if for_landcover:
                patches = [mpatches.Patch(color=cmap(norm(u)), label=lc_map_names[lc_map_inverted[u]]) for u in np.unique(y_pred[idx])]
                plt.legend(handles=patches)
            plt.axis('on')
            plt.grid()

    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()



class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, channels, reduction=16, activation="relu"):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)


class SE_BlockV2(nn.Module):
    # The is a custom implementation of the ideas presented in the paper:
    # https://www.sciencedirect.com/science/article/abs/pii/S0031320321003460
    def __init__(self, channels, reduction=16, activation="relu"):
        super(SE_BlockV2, self).__init__()

        self.channels = channels
        self.reduction = reduction
        self.activation = get_activation(activation)
   
        self.fc_spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(channels, channels, kernel_size=2, stride=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.fc_reduction = nn.Linear(in_features=channels * (4 * 4), out_features=channels // self.reduction)
        self.fc_extention = nn.Linear(in_features=channels // self.reduction , out_features=channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        identity = x
        x = self.fc_spatial(identity)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_reduction(x)
        x = self.activation(x)
        x = self.fc_extention(x)
        x = self.sigmoid(x)
        x = x.reshape(x.size(0), x.size(1), 1, 1)

        return x


class SE_BlockV3(nn.Module):
    """ Squeeze and Excitation block with spatial and channel attention. """
    def __init__(self, channels, reduction_c=2, reduction_s=8, activation="relu", norm="batch", first_layer=False):
        super(SE_BlockV3, self).__init__()

        self.channels = channels
        self.first_layer = first_layer
        self.reduction_c = reduction_c if not first_layer else 1
        self.reduction_s = reduction_s
        self.activation = get_activation(activation)
   
        self.fc_pool = nn.AdaptiveAvgPool2d(reduction_s)
        self.fc_conv = nn.Conv2d(self.channels, self.channels, kernel_size=2, stride=2, groups=self.channels, bias=False)
        self.fc_norm = get_normalization(norm, self.channels)

        self.linear1 = nn.Linear(in_features=self.channels * (reduction_s // 2 * reduction_s // 2), out_features=self.channels // self.reduction_c)
        self.linear2 = nn.Linear(in_features=self.channels // self.reduction_c, out_features=self.channels)

        self.activation_output = nn.Softmax(dim=1) if first_layer else nn.Sigmoid()


    def forward(self, x):
        identity = x

        x = self.fc_pool(x)
        x = self.fc_conv(x)
        x = self.fc_norm(x)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        if self.first_layer:
            x = self.activation_output(x) * x.size(1)
        else:
            x = self.activation_output(x)
            
        x = identity * x.reshape(x.size(0), x.size(1), 1, 1)

        return x


def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU6(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.ReLU6):
        return activation_name

    elif activation_name == "gelu":
        return nn.GELU()
    elif isinstance(activation_name, torch.nn.modules.activation.GELU):
        return activation_name

    elif activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.LeakyReLU):
        return activation_name

    elif activation_name == "prelu":
        return nn.PReLU()
    elif isinstance(activation_name, torch.nn.modules.activation.PReLU):
        return activation_name

    elif activation_name == "selu":
        return nn.SELU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.SELU):
        return activation_name

    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif isinstance(activation_name, torch.nn.modules.activation.Sigmoid):
        return activation_name

    elif activation_name == "tanh":
        return nn.Tanh()
    elif isinstance(activation_name, torch.nn.modules.activation.Tanh):
        return activation_name

    elif activation_name == "mish":
        return nn.Mish()
    elif isinstance(activation_name, torch.nn.modules.activation.Mish):
        return activation_name
    else:
        raise ValueError(f"activation must be one of leaky_relu, prelu, selu, gelu, sigmoid, tanh, relu. Got: {activation_name}")


def get_normalization(normalization_name, num_channels, num_groups=32, dims=2):
    if normalization_name == "batch":
        if dims == 1:
            return nn.BatchNorm1d(num_channels)
        elif dims == 2:
            return nn.BatchNorm2d(num_channels)
        elif dims == 3:
            return nn.BatchNorm3d(num_channels)
    elif normalization_name == "instance":
        if dims == 1:
            return nn.InstanceNorm1d(num_channels)
        elif dims == 2:
            return nn.InstanceNorm2d(num_channels)
        elif dims == 3:
            return nn.InstanceNorm3d(num_channels)
    elif normalization_name == "layer":
        return LayerNorm(num_channels)
    elif normalization_name == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization_name == "bcn":
        if dims == 1:
            return nn.Sequential(
                nn.BatchNorm1d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 2:
            return nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 3:
            return nn.Sequential(
                nn.BatchNorm3d(num_channels),
                nn.GroupNorm(1, num_channels)
            )    
    elif normalization_name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"normalization must be one of batch, instance, layer, group, none. Got: {normalization_name}")

