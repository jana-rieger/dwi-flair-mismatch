import tensorflow as tf
from tensorflow.keras.layers import Conv3D

from groupy.gconv.tensorflow_gconv.splitgconv3d import gconv3d, gconv3d_util


class GroupConv3D(Conv3D):

    def __init__(self,
                 ksize,
                 in_channels,
                 out_channels,
                 h_input='Z3',
                 h_output='O',
                 strides=(1, 1, 1),
                 padding='same',
                 data_format='channels_last',
                 use_bias=False,
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(GroupConv3D, self).__init__(
            filters=None,
            kernel_size=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            **kwargs)
        self.ksize = ksize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_input = h_input
        self.h_output = h_output
        self.gconv_indices = None
        self.gconv_shape_info = None
        self.w_shape = None
        self.w = None

        if use_bias is True:
            raise ValueError('This Version of GConv3D not supporting bias use yet'
                             'Please use it under BatchNorm or any other Normalization Method')

        # modify strides
        self._strides = (1,) + self.strides + (1,)
        # self._padding = _preprocess_padding(self.padding)

        # modify padding
        if padding == 'same':
            self._padding = 'SAME'
        elif padding == 'valid':
            self._padding = 'VALID'
        else:
            raise ValueError('Invalid Padding Value'
                             'Valid values are "same" or "valid')
        # modify data_format
        if self.data_format == 'channels_last':
            self.channel_axis = -1
            self.d_format = 'NHWC'
        else:
            raise ValueError('Currently only NHWC data_format is supported')

    def build(self, input_shape):
        if len(input_shape) < 5:
            raise ValueError('Inputs to `GConv3D` should have rank 5. '
                             'Received input shape:', str(input_shape))

        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`GConv3D` '
                             'should be defined. Found `None`.')

        # self.in_channels = int(input_shape[channel_axis])

        self.gconv_indices, self.gconv_shape_info, self.w_shape = gconv3d_util(h_input=self.h_input,
                                                                               h_output=self.h_output,
                                                                               in_channels=self.in_channels,
                                                                               out_channels=self.out_channels,
                                                                               ksize=self.ksize)
        # print(self.w_shape)
        self.w = self.add_weight(shape=self.w_shape,
                                 initializer=self.kernel_initializer,
                                 name='filters',
                                 regularizer=self.kernel_regularizer, trainable=True)

    def call(self, inputs, training=None):

        outputs = gconv3d(input=inputs, filter=self.w, strides=self._strides, padding=self._padding,
                          gconv_indices=self.gconv_indices, gconv_shape_info=self.gconv_shape_info,
                          data_format=self.d_format, name=self.name)
        return outputs

    def get_config(self):
        config = super(GroupConv3D, self).get_config()
        config.pop('filters')
        config.pop('kernel_size')
        config['ksize'] = self.ksize
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        config['h_input'] = self.h_input
        config['h_output'] = self.h_output
        return config
