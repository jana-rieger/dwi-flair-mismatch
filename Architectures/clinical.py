from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import tensorflow as tf


def ClinicalNN(input_shape,
               model_name,
               depths_of_layers=None,
               include_top=False,
               classes=2,
               kernel_regularizer=None,
               bias_regularizer=None,
               dropout_rate=0,
               final_dropout=0):

    if depths_of_layers is None:
        depths_of_layers = [128, 256, 512]

    # Define input
    inputs = layers.Input(input_shape, name=model_name)
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=model_name + '_input/bn')(inputs)

    # Hidden layers
    for i, units in enumerate(depths_of_layers):
        x = layers.Dense(units, activation=tf.nn.relu, name=model_name + '_dense_' + str(i),
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         bias_initializer=tf.keras.initializers.Constant(0.))(x)
        x = layers.Dropout(rate=dropout_rate, name=model_name + '_dense_' + str(i) + '/drop')(x)

    if final_dropout != 0:
        x = layers.Dropout(rate=final_dropout, name=model_name + '_final_dropout')(x)

    if include_top:
        # Final softmax
        x = layers.Dense(classes, activation=tf.nn.softmax, name=model_name + '_softmax')(x)

    # Create model
    model = Model(inputs, x, name=model_name)

    return model
