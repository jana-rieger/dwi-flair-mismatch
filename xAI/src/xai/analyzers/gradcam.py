'''
GradCAM algorithm.

Paper:
[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

Inspired by:
* https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
* https://github.com/keisen/tf-keras-vis
    - only checks for tensorflow.python.keras.layers.convolutional.Conv
* https://github.com/sicara/tf-explain
    - only for 2D images
    - generates a grid with the original image, can't extract the maps themselves
    - class_index is the same for the entire batch of input images
    - doesn't relu the generated heatmap
'''

from typing import List, Callable
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K

from xAI.src.xai.utils.heatmap import relu_filter
from xAI.src.xai.utils.heatmap import zoom

class GradCAM():
    '''Implements GradCAM algorithm.'''

    def __init__(self,
                 model: tf.keras.Model,
                 loss_func: Callable,
                 ndim: int,
                 input_index: int = 0,
                 layer_name: str = None,
                 guided_grads: bool = True,
                 norm_grads: bool = False,
                 filter_values: bool = True,
                 tag: str = ''):
        '''
        Args:
            model (tf.keras.Model): Keras model.
            loss_func (Callable): Loss function.
            ndim (int): Number of input dimentions (4 for 2D images, 5 for 3D images).
            input_index (int): Input index for each a map is computed.
                               Always 0 for single modal (single-input) models.
            layer_name (str): Name of a target layer. If None, the first 5D layer
                              from the top of the model is returned (Default: None).
            guided_grads (bool): Whether to use guided or raw gradients (Default: True).
            norm_grads (bool): Whether to apply the L2 norm on the gradients (Default: False).
            filter_values (bool): Whether to filter negative values off
                                  of the generated heatmap (Default: True).
            tag (str): Set a tag to distinguish various instances of the analyzer with
                       different parameters (Default: ''). Otherwise, previous results
                       will be overwritten.
        '''

        def find_target_layer(model: tf.keras.Model, layer_name: str, ndim: int) -> str:
            '''
            Find a target output layer.

            Args:
                model (tf.keras.Model): Keras model.
                layer_name (str): Short layer name to be found, e.g., `Conv`, `Pool`. Case insensitive.
                                  If None, the first 5D layer from the top is searched.
                ndim (int): Number of input dimentions (4 for 2D images, 5 for 3D images).

            Returns:
                str: Full target layer name.
            '''

            if layer_name is None:
                # find closest 5D layer to the end of the network
                for layer in reversed(model.layers):
                    if len(layer.output_shape) == ndim:
                        return layer.name
            else:
                layer_name = layer_name.lower().split()
                for layer in reversed(model.layers):
                    if all([word in layer.name.lower() for word in layer_name]):
                        return layer.name

            raise ValueError("Could not find the layer.")

        target_layer_name = find_target_layer(model, layer_name, ndim)

        self.grad_model = Model(inputs=model.inputs,
                                outputs=[model.get_layer(target_layer_name).output, model.output])

        self.loss_func = loss_func
        self.input_index = input_index
        self.guided_grad = guided_grads
        self.norm_grads = norm_grads
        self.filter_values = filter_values

        # for public use
        self.name = 'gradcam' if tag == '' else 'gradcam' + '_' + tag

    def analyze(self, inputs: List[np.ndarray]) -> np.ndarray:
        '''
        Computes GradCAM heatmap.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            np.ndaray: Batch of generated heatmaps for each input image of shape (batch_size, H, W, D).
        '''

        target_layer_outputs, grads = self._compute_grads(inputs)

        heatmaps = self._compute_weighted_output(target_layer_outputs, grads)

        if self.filter_values:
            heatmaps = [relu_filter(heatmap) for heatmap in heatmaps]

        # resize heatmaps to the original image
        heatmaps = [zoom(heatmap.numpy(), inputs[self.input_index]) for heatmap in heatmaps]

        heatmaps = np.stack(heatmaps, axis=0)

        return heatmaps

    def _compute_grads(self, inputs: List[np.ndarray]) -> tuple:
        '''
        Computes guided gradients and target layer outputs.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Target layer outputs and guided gradients
                                         of shape (batch_size, H, W, D, Nf).
        '''

        with tf.GradientTape() as tape:
            inputs = [tf.cast(entry, tf.float32) for entry in inputs]
            tape.watch(inputs)
            target_layer_outputs, predictions = self.grad_model(inputs)
            loss = self.loss_func(predictions, axis=1)

        grads = tape.gradient(loss, target_layer_outputs)

        if self.guided_grad:
            # compute guided grads
            cast_outputs = tf.cast(target_layer_outputs > 0, tf.float32)
            cast_grads = tf.cast(grads > 0, tf.float32)
            grads = cast_outputs * cast_grads * grads

        if self.norm_grads:
            # normalize grads using L2 norm
            grads = K.l2_normalize(grads, axis=np.arange(grads.ndim)[1:-1])

        return target_layer_outputs, grads

    def _compute_weighted_output(self, outputs: tf.Tensor, grads: tf.Tensor) -> tf.Tensor:
        '''
        Weighting of the ouputs of feature maps with respect to the average of gradients.

        Args:
            outputs (tf.Tensor): Target layer outputs of shape (batch_size, H, W, D, Nf)
                                 where N, W, D are the dimensions of a 3D image
                                 and Nf is the number of feature maps.
            grads (tf.Tensor): Guided grads of shape (batch_size, H, W, D, Nf).

        Returns:
            tf.Tensor: Weighted output of shape (batch_size, H, W, D).
        '''

        # define axes: exclude the last one,
        # that is the feature maps dimension
        axes = np.arange(grads.ndim)[1:-1]
        # calculate grad means over these axes, shape (batch_size, 1, 1, 1, Nf)
        weights = tf.reduce_mean(grads, axis=axes, keepdims=True)
        # perform weighting and sum over the last dim (feature maps)
        heatmap = tf.reduce_sum(tf.multiply(weights, outputs), axis=-1)

        return heatmap
