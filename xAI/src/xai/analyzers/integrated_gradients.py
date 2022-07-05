'''
Integrated Gradients algorithm.

Paper:
[Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)

Inspired by:
* [Software and application patterns for explanation methods](https://arxiv.org/abs/1904.04734)
* https://github.com/sicara/tf-explain/blob/master/tf_explain/core/integrated_gradients.py
- calculations differ from the formular:
    - mean over steps instead of sum
    - no (x-x_ref) multiplyer
'''

from typing import Callable, List
import numpy as np
import tensorflow as tf

class IntegratedGradients(object):
    '''Implements Integrated Gradients algorithm.'''

    def __init__(self,
                 model: tf.keras.Model,
                 loss_func: Callable,
                 steps: int = 16,
                 tag: str = ''):
        '''
        Args:
            model (tf.keras.Model): Keras model.
            loss (Callable): Loss function.
            steps (int): Number of steps in approximating the integral.
            tag (str): Set a tag to distinguish various instances of the analyzer with
                       different parameters (Default: ''). Otherwise, previous results
                       will be overwritten.
        '''

        self.model = model
        self.loss_func = loss_func
        self.steps = steps

        # for public use
        self.name = 'intgrads' if tag == '' else 'intgrads' + '_' + tag

    def analyze(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Computes Integrated Gradients heatmap.

        Args:
            inputs (List[np.ndarray]): Input image of shape (batch_size, H, W, D, color_channels).

        Returns:
            List[np.ndarray]: Batch of generated heatmaps for each input image
                              of shape (batch_size, H, W, D, color_channels).
        '''

        int_grads = self._compute_integral(inputs)

        # process gradients
        int_grads = [np.abs(heatmaps) for heatmaps in int_grads]

        return int_grads

    def _compute_integral(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            List[np.ndarray]: Integrated gradients of shape (batch_size, H, W, D, color_channels).
        '''

        # TODO input indeces
        iminputs = inputs[0:2] if len(inputs) > 1 else inputs # imaging inputs
        numinputs = inputs[2:] if len(inputs) > 1 else []     # numeric inputs

        # take a black image as a reference,
        # each entry in the list corresponds to each model's imaging input,
        # numeric data stays untouched
        baselines = [np.zeros_like(iminput) for iminput in iminputs]
        grad_sums = [np.zeros_like(iminput) for iminput in iminputs]

        # sum the gradients along the path from x_ref to x
        for step in range(self.steps):
            cur_iminputs = [baseline + (images - baseline) * step / (self.steps - 1)
                            for baseline, images in zip(baselines, iminputs)]
            # a list of grads of the model's inputs
            input_grads = self._compute_grads(cur_iminputs + numinputs)
            # last numeric data grads in input_grads list will be ignored by zip
            grad_sums = [grad_sum+input_grad.numpy() for grad_sum, input_grad in zip(grad_sums, input_grads)]

        int_grads = [grad_sum * (images - baseline)
                     for baseline, images, grad_sum in zip(baselines, iminputs, grad_sums)]

        return int_grads

    def _compute_grads(self, inputs: List[np.ndarray]) -> List[tf.Tensor]:
        '''
        Computes the gradient of the loss function w.r.t. to the input.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            List[tf.Tensor]: List of gradients of shape (batch_size, H, W, D, color_channels)
                         and numeric data (if available) for each model's input.
        '''

        with tf.GradientTape() as tape:
            inputs = [tf.cast(entry, tf.float32) for entry in inputs]
            tape.watch(inputs)
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, axis=1)

        grads = tape.gradient(loss, inputs)

        return grads
