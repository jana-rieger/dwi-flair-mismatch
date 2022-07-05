'''
GradCAM++ algorithm.

Paper:
[Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/abs/1710.11063)

Inspired by:
* https://github.com/keisen/tf-keras-vis
    - only checks for tensorflow.python.keras.layers.convolutional.Conv
    - why sum of exp score values?
'''

from typing import Callable, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K

from xAI.src.xai.utils.heatmap import relu_filter
from xAI.src.xai.utils.heatmap import zoom

class GradCAMPP():
    '''Implements GradCAM++ algorithm.'''

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
                layer_name (str): Short layer name to be found. 
                                  E.g., `Conv`, `Pool`. Case insensitive.
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

        self.grad_model = Model(inputs=[model.inputs],
                                outputs=[model.get_layer(target_layer_name).output, model.output])

        self.loss_func = loss_func
        self.input_index = input_index
        self.guided_grad = guided_grads
        self.norm_grads = norm_grads
        self.filter_values = filter_values

        # for public use
        self.name = 'gradcampp' if tag == '' else 'gradcampp' + '_' + tag

    def analyze(self, inputs: List[np.ndarray]) -> np.ndarray:
        '''
        Computes GradCAM++ heatmap.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            np.ndaray: Batch of generated heatmaps for each input image of shape (batch_size, H, W, D).
        '''

        target_layer_outputs, grads, score = self._compute_grads(inputs)

        heatmaps = self._compute_weighted_output(target_layer_outputs, grads, score)

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

        score = tf.reduce_sum(K.exp(loss))
        # first_derivative = score * grads
        # second_derivative = first_derivative * grads
        # third_derivative = second_derivative * grads

        return target_layer_outputs, grads, score

    def _compute_weighted_output(self, outputs: tf.Tensor, grads: tf.Tensor, score: tf.Tensor) -> tf.Tensor:
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

        first_derivative = score * grads
        second_derivative = first_derivative * grads
        third_derivative = second_derivative * grads

        global_sum = tf.reduce_sum(outputs, axis=np.arange(grads.ndim)[1:-1], keepdims=True)

        alpha_denom = 2.0 * second_derivative + third_derivative * global_sum
        alpha_denom = alpha_denom + tf.cast((second_derivative == 0.0), second_derivative.dtype)
        alphas = second_derivative / alpha_denom

        alpha_normalization_constant = tf.reduce_sum(alphas,
                                                     axis=np.arange(alphas.ndim)[1:-1],
                                                     keepdims=True)
        alpha_normalization_constant = alpha_normalization_constant + tf.cast(
                                       (alpha_normalization_constant == 0.0), alpha_normalization_constant.dtype)
        alphas = alphas / alpha_normalization_constant

        weights = relu_filter(first_derivative) if self.filter_values else first_derivative

        deep_linearization_weights = weights * alphas
        deep_linearization_weights = tf.reduce_sum(deep_linearization_weights,
                                                   axis=np.arange(deep_linearization_weights.ndim)[1:-1],
                                                   keepdims=True)

        heatmap = tf.reduce_sum(deep_linearization_weights * outputs, axis=-1)

        return heatmap










    # def analyze(self, images: np.ndarray) -> np.ndarray:
    #     '''
    #     Computes GradCAM heatmap.

    #     Args:
    #         images (np.ndarray): Input images of shape (batch_size, H, W, D, color_channels).

    #     Returns:
    #         np.ndaray: Batch of generated heatmaps for each input image of shape (batch_size, H, W, D).
    #     '''

    #     target_layer_outputs, grads = self._compute_grads(images)




    #     # compute heatmaps for each sample over batch dim
    #     heatmaps = [self._compute_weighted_output(o, g) for o, g in zip(target_layer_outputs, grads)]

    #     # resize heatmaps to the original image
    #     heatmaps = [zoom(heatmap.numpy(), images) for heatmap in heatmaps]

    #     if self.filter_values:
    #         heatmaps = [relu_filter(heatmap) for heatmap in heatmaps]

    #     heatmaps = np.stack(heatmaps, axis=0)

    #     return heatmaps

    # def _compute_grads(self, images: np.ndarray) -> tuple:
    #     '''
    #     Computes guided gradients and target layer outputs.

    #     Args:
    #         images (np.ndarray): Input images of shape (batch_size, H, W, D, color_channels).

    #     Returns:
    #         tuple<tf.Tensor, tf.Tensor>: Target layer outputs and guided gradients
    #                                      of shape (batch_size, H, W, D, Nf).
    #     '''

    #     with tf.GradientTape() as tape:
    #         inputs = tf.cast(images, tf.float32)
    #         tape.watch(inputs)
    #         target_layer_outputs, predictions = self.grad_model(inputs)
    #         losses = self.loss_func(predictions, axis=1)

    #     grads = tape.gradient(losses, target_layer_outputs)

    #     if self.guided_grad:
    #         # compute guided grads
    #         cast_outputs = tf.cast(target_layer_outputs > 0, tf.float32)
    #         cast_grads = tf.cast(grads > 0, tf.float32)
    #         grads = cast_outputs * cast_grads * grads

    #     if self.norm_grads:
    #         # normalize grads using L2 norm
    #         grads = K.l2_normalize(grads)


    #     score = K.exp(losses)
    #     first_derivative = score * grads
    #     second_derivative = first_derivative * grads
    #     third_derivative = second_derivative * grads

        

    #     return target_layer_outputs, grads

    # def _compute_weighted_output(self, outputs: tf.Tensor, grads: tf.Tensor) -> tf.Tensor:
    #     '''
    #     Weighting of the ouputs of feature maps with respect to the average of gradients.

    #     Args:
    #         outputs (tf.Tensor): Target layer outputs of shape (H, W, D, Nf)
    #                              where N, W, D are the dimensions of a 3D image
    #                              and Nf is the number of feature maps.
    #         grads (tf.Tensor): Guided grads of shape (H, W, D, Nf).

    #     Returns:
    #         tf.Tensor: Weighted output of shape (H, W, D).
    #     '''

    #     # define axes: exclude the last one,
    #     # that is the feature maps dimension
    #     axes = np.arange(tf.rank(grads))[:-1]
    #     # calculate grad means over these axes, shape (Nf,)
    #     weights = tf.reduce_mean(grads, axis=axes)
    #     # perform weighting and sum over the last dim (feature maps)
    #     heatmap = tf.reduce_sum(tf.multiply(weights, outputs), axis=-1)

    #     return heatmap
