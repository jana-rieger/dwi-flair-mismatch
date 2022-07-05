'''
Occlusion sensitivity algorithm.

Paper:
[Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)

Inspired by:
* [Software and application patterns for explanation methods](https://arxiv.org/abs/1904.04734)
* https://github.com/sicara/tf-explain/blob/master/tf_explain/core/occlusion_sensitivity.py
    - supports only 2D images.
    - only for class scores.
'''

from typing import Callable, Iterable
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

class OcclusionSensitivity(object):
    '''Implements Occlusion Sensitivity algorithm.'''

    def __init__(self,
                 model: tf.keras.Model,
                 loss_func: Callable,
                 occluding_vortex: int = 0,
                 patch_size: Iterable = (40, 40, 10),
                 tag: str = ''):
        '''
        Args:
            model (tf.keras.Model): Keras model.
            loss_func (Callable): Loss function. If None, the maximum value
                                  in the output layer is returned.
            occluding_vortex (int): Vortex value to occlude with.
            patch_size (Iterable): Size of a patch along each dimension.
            tag (str): Set a tag to distinguish various instances of the analyzer with
                       different parameters (Default: ''). Otherwise, previous results
                       will be overwritten.
        '''

        self.model = model
        self.loss_func = loss_func
        self.occluding_vortex = occluding_vortex
        self.patch_size = patch_size

        # for public use
        self.name = 'occlusion' if tag == '' else 'occlusion' + '_' + tag

    def analyze(self, images: np.ndarray) -> np.ndarray:
        '''
        Computes occlusion sensitivity heatmap.

        Args:
            images (np.ndarray): Input images of shape (batch_size, H, W, D, color_channels).

        Returns:
            np.ndaray: Batch of generated heatmaps for each input image
                       of shape (batch_size, H, W, D, color_channels).
        '''

        sensitivity_maps = [self._compute_occlusion(image).numpy() for image in images]

        # concatinate over a batch dimension
        sensitivity_maps = np.concatenate(sensitivity_maps, axis=0)

        return sensitivity_maps

    def _compute_occlusion(self, image: np.ndarray) -> tf.Tensor:
        '''
        Computes occlusion sensitivity map for a given image.

        Args:
            images (np.ndarray): Input images of shape (H, W, D, color_channels).

        Returns:
            tf.Tensor: Batch of generated heatmaps for each input image of shape (1, H, W, D, color_channels).
        '''

        # expand batch dim for the model
        image = np.expand_dims(image, axis=0)

        sensitivity = np.zeros_like(image)

        # occlude patch by patch and calculate activation for each patch
        isize, jsize, ksize = self.patch_size
        for i in range(0, image.shape[1], isize):
            for j in range(0, image.shape[2], jsize):
                for k in range(0, image.shape[3], ksize):
                    # create image with the patch occluded
                    patched_image = image.copy()
                    patched_image[:, i:i+isize, j:j+jsize, k:k+ksize, :] = self.occluding_vortex

                    # store activation of occluded image
                    prediction = self.model(patched_image)
                    # score = prediction[target_class]
                    score = self.loss_func(prediction, axis=1)
                    sensitivity[:, i:i+isize, j:j+jsize, k:k+ksize, :] = score

        # normalize with initial activation value
        output = self.model(image)
        score = self.loss_func(output, axis=1)
        sensitivity = score - sensitivity

        return sensitivity
