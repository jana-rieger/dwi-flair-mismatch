'''
SmoothGrad algorithm.

 Paper:
 [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)

 Inspired by:
* https://github.com/keisen/tf-keras-vis/blob/master/tf_keras_vis/saliency.py
    - when keep_dims=False, takes max over channel dim
    - modifies (takes abs) gradients for each noise image instead of after calculating average gradients
* https://github.com/sicara/tf-explain/blob/master/tf_explain/core/smoothgrad.py
    - requires class index
'''

from typing import Callable, List
import tensorflow as tf
import numpy as np

class SmoothGrad():
    '''Implements SmoothGrad algorithm.'''

    def __init__(self,
                 model: tf.keras.Model,
                 loss_func: Callable,
                 num_samples: int = 5,
                 noise: float = 1.0,
                 tag: str = ''):
        '''
        Args:
            model (tf.keras.Model): Keras model.
            loss (Callable): Loss function.
            num_samples (int): Number of noise samples to generate.
            noise (float): Standard deviation of the normal distribution.
            tag (str): Set a tag to distinguish various instances of the analyzer with
                       different parameters (Default: ''). Otherwise, previous results
                       will be overwritten.
        '''

        self.model = model
        self.loss_func = loss_func
        self.num_samples = num_samples
        self.noise = noise

        # for public use
        self.name = 'smoothgrad' if tag == '' else 'smoothgrad' + '_' + tag

    def analyze(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Computes SmoothGrad heatmap.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            List[np.ndarray]: List of heatmaps of shape (batch_size, H, W, D, color_channels)
                              and numeric data (if available) for each model's input.
        '''

        noisy_inputs = self._generate_noisy_inputs(inputs)

        smooth_grads = self._compute_smoothed_gradient(noisy_inputs)

        # process gradients
        smooth_grads = [np.abs(heatmaps) for heatmaps in smooth_grads]

        return smooth_grads

    def _generate_noisy_inputs(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Generate noisy images for a given image.

        Args:
            inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
                                       and numeric data if available.

        Returns:
            List[np.ndarray]: List of heatmaps of shape (batch_size*num_samples, H, W, D, color_channels)
                              and numeric data (if available) for each model's input.
        '''

        noisy_inputs = []
        for current_input in inputs:
            # make copies of the input
            # shape (batch_size*num_samples, H, W, D, color_channels)
            repeated_input = np.repeat(current_input, self.num_samples, axis=0)

            # generate noise
            noise = np.random.normal(loc=0, scale=self.noise, size=repeated_input.shape).astype(np.float32)

            noisy_inputs.append(repeated_input + noise)

        return noisy_inputs

    def _compute_smoothed_gradient(self, noisy_inputs: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Computes smoothed gradients for a given image.

        Args:
            noisy_inputs (List[np.ndarray]): Input data of images of shape
                                             (batch_size*num_samples, H, W, D, color_channels)
                                             and numeric data if available.

        Returns:
            List[np.ndarray]: List of heatmaps of shape (batch_size, H, W, D, color_channels)
                              and numeric data (if available) for each model's input.
        '''

        # reshape to (num_samples, batch_size, H, W, D, color_channels)
        noisy_inputs = [noisy_input.reshape((self.num_samples, -1, *noisy_input.shape[1:]))
                        for noisy_input in noisy_inputs]

        # generating samples for 3D images can take up a lot of memory
        # requiring in total batch_size*num_samples gradients calculations at once
        # to save memory, compute as a cumulative sum, shape (batch_size, H, W, D, color_channels)
        grad_sums = [np.zeros(noisy_input.shape[1:]) for noisy_input in noisy_inputs]
        for sample_num in range(self.num_samples):

            # get a list of inputs, shape (batch_size, H, W, D, color_channels)
            sample_inputs = [noisy_input[sample_num] for noisy_input in noisy_inputs]

            sample_grads = self._compute_grads(sample_inputs)

            grad_sums = [grad_sum + sample_grad.numpy() for grad_sum, sample_grad in zip(grad_sums, sample_grads)]

        smooth_grads = [grad_sum / self.num_samples for grad_sum in grad_sums]

        return smooth_grads

    def _compute_grads(self, inputs: List[np.ndarray]) -> List[tf.Tensor]:
        '''
        Computes an average gradient of the gradients of the loss function w.r.t. to the inputs.

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
