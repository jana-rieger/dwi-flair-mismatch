'''
Saliency map algorithm.

Paper:
[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)

Inspired by:
* https://github.com/keisen/tf-keras-vis/blob/master/tf_keras_vis/saliency.py
    - when keep_dims=False, takes max over channel dim
* https://github.com/sicara/tf-explain/blob/master/tf_explain/core/vanilla_gradients.py
    - requires class index.
'''

from typing import Callable, List
import tensorflow as tf
import numpy as np

class VanillaGradients():
    '''Implements Saliency algorithm.'''

    def __init__(self, model: tf.keras.Model, loss_func: Callable, tag: str = ''):
        '''
        Args:
            model (tf.keras.Model): Keras model.
            loss (Callable): Loss function.
            tag (str): Set a tag to distinguish various instances of the analyzer with
                       different parameters (Default: ''). Otherwise, previous results
                       will be overwritten.
        '''

        self.model = model
        self.loss_func = loss_func

        # for public use
        self.name = 'vangrads' if tag == '' else 'vangrads' + '_' + tag

    def analyze(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Computes Saliency heatmap.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            List[np.ndarray]: List of heatmaps of shape (batch_size, H, W, D, color_channels)
                              and numeric data (if available) for each model's input.
        '''

        grads = self._compute_grads(inputs)

        # process gradients
        grads = [np.abs(heatmaps) for heatmaps in grads]

        return grads

    def _compute_grads(self, inputs: List[np.ndarray]) -> List[tf.Tensor]:
        '''
        Computes the gradient of the loss function w.r.t. to the input.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            List[np.ndarray]: List of gradients of shape (batch_size, H, W, D, color_channels)
                              and numeric data (if available) for each model's input.
        '''

        with tf.GradientTape() as tape:
            inputs = [tf.cast(entry, tf.float32) for entry in inputs]
            tape.watch(inputs)
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, axis=1)

        grads = tape.gradient(loss, inputs)

        return grads
