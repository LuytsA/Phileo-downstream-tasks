import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
import random
import matplotlib.pyplot as plt
import gc
gc.collect(2)
import numpy as np
import buteo as beo
import os


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


def visualise(x, y, y_pred=None, images=5, channel_first=False, vmin=0, vmax=1, save_path=None):
    rows = images
    if y_pred is None:
        columns = 2
    else:
        columns = 3
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

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
        plt.imshow(y[idx], vmin=vmin, vmax=vmax, cmap='magma')
        plt.axis('on')
        plt.grid()

        if y_pred is not None:
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(y_pred[idx], vmin=vmin, vmax=vmax, cmap='magma')
            plt.axis('on')
            plt.grid()

    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


