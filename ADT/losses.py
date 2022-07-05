import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K, layers

from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from termcolor import colored

# Compatible with tensorflow backend

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


class CombinedCrossentropy(tf.keras.losses.Loss):
    def __init__(self,
                 alpha=None,
                 dich_class_threshold=None,
                 num_classes=None,
                 name='comb_crossent', **kwargs):
        """
        Combined crossentropy loss. Weighted sum of binary and categorical crossentropy.

        :param alpha: Weight for categorical crossentropy between 0 and 1. The binary crossentropy is weighted by
            (1 - alpha).
        :param dich_class_threshold: Threshold class for dichotomization of the multiclass output.
            E.g. threshold = 2, classes = [0, 1, 2, 3, 4, 5, 6] -> dichotomized class 0 includes classes [0, 1, 2],
            dichotomzed class 1 includes classes [3, 4, 5, 6].
        :param num_classes: Number of classes.

        """
        super(CombinedCrossentropy, self).__init__(name=name, **kwargs)

        self.alpha = alpha
        self.dich_class_threshold = dich_class_threshold
        self.magnitude = np.log(1./2.) / np.log(1./float(num_classes))

    def call(self, y_true, y_pred):
        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # Calculate categorical crossentropy.
        categorical_crossent = tf.scalar_mul(self.magnitude, sparse_categorical_crossentropy(y_true, y_pred))
        categorical_crossent = tf.scalar_mul(self.alpha, categorical_crossent)
        categorical_crossent = tf.reduce_mean(tf.cast(categorical_crossent, tf.float32))

        # Convert multiclass y_true and y_pred to binary.
        # y_true are integers.
        b_y_true = tf.math.greater(y_true, tf.constant([self.dich_class_threshold], dtype=tf.float32))
        b_y_true = tf.cast(b_y_true, tf.float32)
        # y_pred are softmax probabilities.
        b_y_pred = tf.clip_by_value(tf.reduce_sum(y_pred[:, self.dich_class_threshold + 1:], axis=1), clip_value_min=0,
                                    clip_value_max=1)

        # Calculate binary crossentropy.
        binary_crossent = tf.scalar_mul((1 - self.alpha), binary_crossentropy(b_y_true, b_y_pred))
        binary_crossent = tf.reduce_mean(tf.cast(binary_crossent, tf.float32))

        # Sum up both losses.
        return binary_crossent + categorical_crossent


def ae_loss(alpha=None, base_loss=tf.keras.losses.binary_crossentropy,
            reconstruction_loss=tf.keras.losses.mean_squared_error, model_outputs=None,
            magnitude=1):
    """
    :param base_loss: Loss function for the classification/regression output.
    :param reconstruction_loss: Loss function for the autoencoder image reconstruction output.
    :param alpha: Weight for the base loss (classification/regression loss), values between 0 and 1.
                The reconstruction loss is weighted by (1 - alpha).
    :param model_outputs: Output layers of the model.
    :return: Dictionary of loss functions for each output. Dictionary of loss weights for each output.
    """

    losses = {}
    loss_weights = {}

    if alpha < 0 or alpha > 1:
        raise AssertionError

    for out in model_outputs:
        if 'final_output' in out.name:
            losses[out.name.split('/')[0]] = base_loss
            loss_weights[out.name.split('/')[0]] = alpha * magnitude
        elif 'decoder_output' in out.name:
            losses[out.name.split('/')[0]] = reconstruction_loss
            loss_weights[out.name.split('/')[0]] = (1 - alpha) / (len(model_outputs) - 1)
        else:
            raise NameError('Unknown model output: ' + out.name + '. Cannot match loss function.')

    print('LOSSES:', losses)
    print('LOSS WEIGHTS:', loss_weights)

    return losses, loss_weights


if __name__ == '__main__':
    from tensorflow.keras.models import Model

    dummy_input = layers.Input(shape=(192, 192, 50, 2), name='embedding')
    dummy_out1 = layers.Dense(1, activation=tf.nn.sigmoid,
                                       name='final_output_sigmoid')(dummy_input)
    dummy_out2 = layers.Dense(2, activation='linear',
                                       name='dwi_decoder_output_linear')(dummy_input)
    dummy_out3 = layers.Dense(2, activation='linear',
                              name='flair_decoder_output_linear')(dummy_input)
    model = Model(dummy_input, [dummy_out1, dummy_out2, dummy_out3], name='dummy_model')
    model.summary(line_length=150)

    alpha = 0.7
    losses, loss_weights = ae_loss(alpha=alpha, model_outputs=model.outputs)

    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
