'''Heatmaps utilities.'''

from typing import Iterable
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
from tensorflow.keras import backend as K

relu_filter = lambda x: K.relu(x)


def process_color_channel(heatmap: np.ndarray) -> np.ndarray:
    '''
    Squeezes color dimension if #channels equals 1,
    othewise converts the heatmap into grayscale.

    Args:
        heatmap (np.ndarray): Heatmap with color channels.

    Returns:
        np.ndarray: Heatmap with a reduced color dimension.
    '''

    color_channels = heatmap.shape[-1]

    if color_channels == 1:
        heatmap = np.squeeze(heatmap)
    else:  # 2 or 3
        # convert to grayscale
        heatmap = np.sum(heatmap, axis=-1)
        # heatmap = (255*heatmap).astype(np.uint8)

    return heatmap


def zoom(heatmap: np.ndarray, images: np.ndarray) -> np.ndarray:
    '''
    Zooms heatmap to the size of the original image.

    Args:
        heatmap (np.ndarray): Heatmap of shape (H, W, (D)) to be resized.
        images (np.ndarray): Original images with batch and channels dimensions.

    Output:
        np.ndarray: Resized heatmap.
    '''

    image_dims = images.shape[1:-1]  # exclude batch and channels dims
    heatmap_dims = heatmap.shape
    factors = [float(m / n) for m, n in zip(image_dims, heatmap_dims)]

    heatmap = ndimage.zoom(heatmap, factors)

    return heatmap


def filter_extreme(heatmap: np.ndarray, percentile: int = 99) -> np.ndarray:
    '''
    Caps extreme values with 99-percentile.

    Args:
        heatmap (np.ndarray): Heatmap to be filtered.

    Returns:
        np.ndarray: Filtered heatmap.
    '''

    p = np.percentile(heatmap, percentile)

    heatmap[heatmap > p] = p

    return heatmap


def standardize(heatmap: np.ndarray) -> np.ndarray:
    '''
    Performs a standardization of a heatmap.

    Args:
        heatmap (np.ndarray): Heatmap to be standardized.

    Returns:
        np.ndarray: Standardized heatmap.
    '''

    # standardize tensor
    heatmap -= heatmap.mean()
    heatmap /= (heatmap.std() + K.epsilon())

    return heatmap


def normalize(heatmap: np.ndarray, bounds: Iterable = (0., 1.)) -> np.ndarray:
    '''
    By default performs a normalization of a heatmap.
    To scale into a different interval, pass desired bounds.

    Args:
        heatmap (np.ndarray): Heatmap to be scaled.
        bounds (iterable): Scaling interval, default [0,1].

    Returns:
        np.ndarray: Normalized/scaled heatmap.
    '''

    min_value = heatmap.min()
    max_value = heatmap.max()
    lower_bound, upper_bound = bounds

    # normalize
    heatmap = (heatmap - min_value) / (max_value - min_value + K.epsilon())
    # scale to [lower_bound, upper_bound] interval
    heatmap = lower_bound + heatmap * (upper_bound - lower_bound)

    return heatmap


def threshold(heatmap: np.ndarray, level: float = 0.35) -> np.ndarray:
    '''
    Thresholds a heatmap given the level.

    Args:
        heatmap (np.ndarray): Heatmap to thresholded.
        level (float): Threshold point.

    Returns:
        np.ndarray: Heatmap after thresholding.
    '''

    heatmap = (heatmap - level) / (1.0 - level)
    heatmap[heatmap < 0.0] = 0.0

    return heatmap


def convert(heatmap: np.ndarray) -> np.ndarray:
    '''
    Convert a float array into a valid uint8 image.

    Inspired by:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py

    Args:
        heatmap (np.ndarray): Heatmap to be scaled.

    Returns:
        np.ndarray: Standardized heatmap.
    '''

    heatmap = standardize(heatmap)
    # ensure std is 0.25
    heatmap *= 0.25

    # clip to [0, 1]
    heatmap += 0.5
    heatmap = np.clip(heatmap, 0, 1)

    # convert to RGB array
    heatmap *= 255
    heatmap = np.clip(heatmap, 0, 255).astype('uint8')
    return heatmap


def save(heatmap: np.ndarray, nifti_affine: np.ndarray, output_path: str, name_prefix: str, name: str,
         channels: bool):
    '''
    Saves heatmaps into nifti images.

    Args:
        name_prefix (str): Name of an analyzer.
        name (str): Names of an original image.
        heatmap (tf.Tensor): Heatmap to be saved.
        nifti_affine (np.ndarray): Original affine transformations of nifti images.
        output_path (str): Directory for the images to be saved in.
        channels (bool): Whether the heatmap has color channels.
    '''

    # define file name
    file_name = name_prefix + '_' + name
    if channels:
        path_parts = file_name.split('.')
        file_name = path_parts[0] + '+flair'
        for i, p in enumerate(path_parts):
            if i > 0:
                file_name += '.' + p
    file_name = file_name if '.gz' in name else file_name + '.gz'

    nifti_image = nib.Nifti1Image(heatmap, nifti_affine)
    print("Saving to:", output_path, file_name)
    nib.save(nifti_image, os.path.join(output_path, file_name))


def save_npz(heatmap: np.ndarray, output_path: str, name_prefix: str, name: list):
    '''
    Saves heatmaps into nifti images.

    Args:
        name_prefix (str): Name of an analyzer.
        name (str): Names of an original image.
        heatmap (tf.Tensor): Heatmap to be saved.
        output_path (str): Directory for the images to be saved in.
    '''

    # define file name
    file_name = name_prefix + '_' + name
    file_name = file_name if '.npz' in name else file_name + '.npz'

    np.savez_compressed(os.path.join(output_path, file_name), heatmap)
