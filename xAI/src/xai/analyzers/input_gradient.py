'''
Input*Gradient algorithm.

Paper:
[Investigating the influence of noise and distractors on the interpretation of neural networks](https://arxiv.org/abs/1611.07270)
[Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)


Inspired by:
* [Software and application patterns for explanation methods](https://arxiv.org/abs/1904.04734)
* https://github.com/sicara/tf-explain/blob/master/tf_explain/core/gradients_inputs.py
'''

from typing import List, Callable
import numpy as np
import tensorflow as tf

class InputGradient():
    '''Implements Input*Gradient algorithm.'''

    def __init__(self, model: tf.keras.Model, loss_func: Callable, tag: str = ''):
        '''
        Args:
            model (tf.keras.Model): Keras model.
            loss_func (Callable): Loss function.
            tag (str): Set a tag to distinguish various instances of the analyzer with
                       different parameters (Default: ''). Otherwise, previous results
                       will be overwritten.
        '''

        self.model = model
        self.loss_func = loss_func

        # for public use
        self.name = 'insgrads' if tag == '' else 'insgrads' + '_' + tag

    def analyze(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Computes Input*Gradient heatmaps for given inputs.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            List[np.ndarray]: List of heatmaps of shape (batch_size, H, W, D, color_channels)
                              and numeric data (if available) for each model's input.
        '''

        inputs_gradients = self._compute_insgrads(inputs)

        # process gradients
        inputs_gradients = [np.abs(heatmaps) for heatmaps in inputs_gradients]

        return inputs_gradients

    def _compute_insgrads(self, inputs: List[np.ndarray]) -> List[tf.Tensor]:
        '''
        Computes Inputs*Gradients.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data, if available.

        Returns:
            List[List[np.ndarray]]: List of heatmaps of shape (batch_size, H, W, D, color_channels)
                                    and numeric data (if available) for each model's input.
        '''

        with tf.GradientTape() as tape:
            inputs = [tf.cast(entry, tf.float32) for entry in inputs]
            tape.watch(inputs)
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, axis=1)

        grads = tape.gradient(loss, inputs)
        inputs_gradients = [tf.multiply(input, grad) for input, grad in zip(inputs, grads)]

        return inputs_gradients
