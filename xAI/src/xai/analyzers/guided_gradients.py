'''
Guided gradients algorithm.

Paper:
[Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
'''

# from typing import Callable, List
# import tensorflow as tf
# import numpy as np

# class GuidedGradients():
#     '''Implements Saliency algorithm.'''

#     def __init__(self, model: tf.keras.Model, loss_func: Callable, tag: str = ''):
#         '''
#         Args:
#             model (tf.keras.Model): Keras model.
#             loss (Callable): Loss function.
#             tag (str): Set a tag to distinguish various instances of the analyzer with
#                        different parameters (Default: ''). Otherwise, previous results
#                        will be overwritten.
#         '''

#         self.model = model
#         self.loss_func = loss_func

#         # for public use
#         self.name = 'guidedgrads' if tag == '' else 'guidedgrads' + '_' + tag

#     def analyze(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
#         '''
#         Computes Guided Gradients attribution map.

#         Args:
#             inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
#                                        and numeric data if available.

#         Returns:
#             List[np.ndarray]: List of heatmaps of shape (batch_size, H, W, D, color_channels)
#                               and numeric data (if available) for each model's input.
#         '''

#         grads = self._compute_grads(inputs)

#         return grads

#     def _compute_grads(self, inputs: List[np.ndarray]) -> List[tf.Tensor]:
#         '''
#         Computes the gradient of the loss function w.r.t. to the input.

#         Args:
#             inputs (List[np.ndarray]): Input data of images of shape (batch_size, H, W, D, color_channels)
#                                        and numeric data if available.

#         Returns:
#             List[np.ndarray]: List of gradients of shape (batch_size, H, W, D, color_channels)
#                               and numeric data (if available) for each model's input.
#         '''

#         with tf.GradientTape() as tape:
#             inputs = [tf.cast(entry, tf.float32) for entry in inputs]
#             tape.watch(inputs)
#             outputs = self.model(inputs)
#             loss = self.loss_func(outputs, axis=1)

#         grads = tape.gradient(loss, inputs)

#         # compute guided grads
#         cast_outputs = tf.cast(outputs > 0, tf.float32)
#         cast_grads = tf.cast(grads > 0, tf.float32)
#         grads = cast_outputs * cast_grads * grads

#         return grads
