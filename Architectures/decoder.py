import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers


def Decoder(input_layer, output_shape, ksize, decoder_stages, kernel_initializer, kernel_regularizer,
            dropout_rate, modality, from_embedding=True, residuals_list=None):
    """
    :param input_layer: Embedding tensor.
    :param output_shape: Shape of the decoder output without batch size.
    """
    logging.debug('desired output shape: %s', output_shape)
    logging.debug('input layer shape: %s', input_layer.shape)

    kernel_reg = regularizers.l2(kernel_regularizer)
    print('Decoder kernel reg:', kernel_regularizer)

    if from_embedding:
        x = layers.Reshape((1, 1, 1, input_layer.shape[-1]))(input_layer)
    else:
        x = input_layer
    x = layers.Conv3D(filters=input_layer.shape[-1] // 2, kernel_size=ksize, padding='same',
                      kernel_initializer=kernel_initializer, kernel_regularizer=kernel_reg,
                      name=modality + '_decod_bottleneck/conv')(x)
    x = layers.BatchNormalization(axis=-1, name=modality + '_decod_bottleneck/bn')(x)
    x = layers.Activation(activation='relu', name=modality + '_decod_bottleneck/relu')(x)

    up_size = np.asarray(output_shape[:-1]) // (2 ** (decoder_stages - 1)) // x.shape[1:-1]

    for i in range(decoder_stages):
        logging.debug('LEVEL: %s', i)

        logging.debug('before upsampling: %s', x.shape)
        up_size = up_size if i == 0 else 2
        logging.debug('up size: %s', up_size)

        x = layers.UpSampling3D((up_size), name=modality + '_decod_' + str(decoder_stages - i) + '/up')(x)
        logging.debug('after upsampling: %s', x.shape)

        out_channels = x.shape[-1] // 2

        if residuals_list is not None:
            x = tf.concat([residuals_list[-1 - i], x], axis=-1)

        x = layers.Conv3D(filters=out_channels, kernel_size=ksize, padding='same',
                          kernel_initializer=kernel_initializer, kernel_regularizer=kernel_reg,
                          name=modality + '_decod_' + str(decoder_stages - i) + '/conv')(x)
        logging.debug('after convolution: %s', x.shape)

        x = layers.BatchNormalization(axis=-1, name=modality + '_decod_' + str(decoder_stages - i) + '/bn')(x)

        x = layers.Activation(activation='relu', name=modality + '_decod_' + str(decoder_stages - i) + '/relu')(x)

        x = layers.SpatialDropout3D(rate=dropout_rate, name=modality + '_decod_' + str(decoder_stages - i) + '/drop')(x)

    logging.debug('before final layer: %s', x.shape)

    # Final layer.
    # for regression
    # decoder_output = layers.Dense(output_shape[-1], activation='linear',
    #                               name=modality + '_decoder_output_linear')(x)
    decoder_output = layers.Conv3D(filters=output_shape[-1], kernel_size=ksize, padding='same', activation='linear',
                                   kernel_initializer=kernel_initializer, kernel_regularizer=kernel_reg,
                                   name=modality + '_decoder_output_linear')(x)

    logging.debug('final output: %s', decoder_output.shape)

    return decoder_output


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    dummy_input = layers.Input(shape=(192, 192, 48, 2), name='decoder')
    dummy_input2 = layers.Input(shape=(6, 6, 3, 512), name='decoder2')
    output_shape = (96, 96, 48, 2)
    dummy_embedding = layers.GlobalAveragePooling3D()(layers.Dense(256)(dummy_input))
    decoder_stages = 4

    print('DECODER FROM 1D EMBEDDING')
    decoded = Decoder(dummy_embedding, output_shape, ksize=3, decoder_stages=decoder_stages,
                      kernel_initializer='he_normal', kernel_regularizer=None, dropout_rate=0, modality='dwi')

    print('*' * 100)
    print('DECODER FROM 3D LAYER')
    decoded2 = Decoder(dummy_input2, output_shape, ksize=3, decoder_stages=decoder_stages,
                       kernel_initializer='he_normal', kernel_regularizer=None, dropout_rate=0, modality='dwi',
                       from_embedding=False)
