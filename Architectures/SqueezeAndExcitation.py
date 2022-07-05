from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import layers, initializers
from tensorflow.keras.models import Model


def SE_block(input_tensor, name, ratio=4, kernel_regularizer=None, bias_regularizer=None, initializer='he_normal'):
    # Squeeze: Average Pool Spatial Data per Channel into Units
    squeeze = layers.GlobalAveragePooling3D(name=name + '/squeeze')(input_tensor)

    channel_axis = -1
    filters = squeeze.shape[channel_axis]
    squeeze = layers.Reshape(target_shape=(1, 1, 1, filters), name=name + '/reshape')(squeeze)

    # Excite: Learn channels interdependencies through Fully Connected Layers with Nonlinearities
    x = layers.Dense(units=filters // ratio, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                     bias_initializer=initializers.Constant(0.1),
                     name=name + '/excite_relu')(squeeze)

    x = layers.Dense(units=filters, activation='sigmoid', kernel_initializer=initializer,
                     kernel_regularizer=kernel_regularizer, bias_initializer=initializers.Constant(0.1),
                     bias_regularizer=bias_regularizer, name=name + '/excite_sigmoid')(x)

    # x = layers.Reshape(target_shape=(1,1,1,filters), name=name+'/reshape')(x)

    # Multiply: Weight the input channels with the excitations
    x = layers.multiply([x, input_tensor], name=name + '/multiply')

    return x


# TODO on the works
class SE(layers.Layer):
    def __init__(self, name=None, ratio=4, kernel_regularizer=None, bias_regularizer=None, initializer='he_normal', **kwargs):
        super(SE, self).__init__(name, **kwargs)
        self.ratio = ratio
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.initializer = initializer

    def call(self, input_tensor, **kwargs):
        # TODO here
        return SE_block(input_tensor, name=self.name, ratio=self.ratio, kernel_regularizer=self.kernel_regularizer,
                        initializer=self.initializer)

    def get_config(self):
        config = super(SE, self).get_config()
        config.update({"ratio": self.ratio})
        return config


def trynet(input_shape, model_name='trynet'):
    input_image = layers.Input(shape=input_shape)

    x = layers.Conv3D(filters=12, kernel_size=3, strides=2, padding='same', name=model_name + '/conv1')(input_image)

    # x = SE_block(x, name=model_name + '_SE', ratio=4)
    x = SE(name=model_name + '_SE', ratio=4)(x)

    x = layers.Conv3D(filters=24, kernel_size=3, strides=1, padding='same', name=model_name + '/conv2')(x)

    print(x.shape)
    x = layers.GlobalAveragePooling3D(name=model_name + '/final_avg_pool')(x)

    print(x.shape)
    # Create Model

    model = Model(input_image, x, name=model_name)
    model.summary()

    return model


if __name__ == '__main__':
    trynet((8, 8, 8, 1))
