from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from Architectures.GroupConv3D import GroupConv3D
from Architectures.SqueezeAndExcitation import SE_block

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from ADT.norm import GroupNormalization


def norm_act(x, name='norm_act', bias_regularizer=None, dropout_rate=0):
    x = layers.BatchNormalization(axis=-1, name=name + '/bn', beta_regularizer=bias_regularizer,
                                  beta_initializer=Constant(0.1))(x)
    x = layers.Activation(activation='relu', name=name + '/relu')(x)
    if dropout_rate != 0:
        x = layers.SpatialDropout3D(rate=dropout_rate, name=name + '/drop')(x)
    return x


def GConv_Block(x, gconv, ksize, in_channels, out_channels, h_input,
                h_output, kernel_initializer, kernel_regularizer,
                bias_regularizer, dropout_rate,
                name, strides=(1, 1, 1)):

    if gconv is True:
        x = GroupConv3D(ksize=ksize, in_channels=in_channels, strides=strides, out_channels=out_channels,
                        use_bias=False, h_input=h_input, h_output=h_output, kernel_regularizer=kernel_regularizer,
                        kernel_initializer=kernel_initializer, name=name + '/gconv')(x)
    else:
        x = layers.Conv3D(kernel_size=ksize, filters=out_channels, strides=strides, padding='same',
                          use_bias=False, kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer,
                          name=name + '/conv')(x)

    x = norm_act(x, name=name, bias_regularizer=bias_regularizer, dropout_rate=dropout_rate)

    return x


def GroupNet(input_shape,
             model_name='GCNN',
             gconv=True,
             stages=None,
             out_channels=16,
             h_output='D4H',
             kernel_size=3,
             init_strides=(2, 2, 1),
             dropout_rate=0,
             final_dropout=0.,
             kernel_initializer='he_normal',
             kernel_regularizer=None,
             bias_regularizer=None,
             skips=False,
             use_se=True,
             include_top=False):
    """
    architecture similar to http://arxiv.org/abs/1804.04656

    :param input_shape: shape of image tensor
    :param model_name: name of the model. code words used: dwi, flair, clin
    :param gconv: Boolean, True for G-Convs, False for plain Convs. Direct Comparison
    :param stages:  array containing how many convolutional blocks there is on each stage, from stage 2
                    stage 1 is not included here, this is always set to one convolutional block
                    spatial-size get reduced by half at the beginning of each stage by Max-Pooling
                    number of output feature channels gets doubled by the first gconv on SOME stages (2 and 4 for now)
                    (not in all stages because of MEMORY Constrains)
                    example: stages=[2,5,10]
                    number of gconv-layers in
                        stage 1 = 1
                        stage 2 = 2
                        stage 3 = 5
                        stage 4 = 10
                        stage 5 = 0
    :param out_channels: initial number of output feature channels on stage 1. Determines the whole width of the net
    :param h_output: (String) name of the symmetry group: Z3, C4H, D4H, O, OH -- use one
    :param kernel_size: size of the 3D convolutional kernel
    :param init_strides: strides on the stage 1 convolution
    :param dropout_rate: spatial-dropout rate use only on odd stages after used in http://arxiv.org/abs/1804.04656
    :param final_dropout: spatial-dropout only on the output embedding, only for experimental use
    :param kernel_initializer: (String) initializing distribution for kernel weights
    :param kernel_regularizer: kernel weight decay Keras Object
    :param bias_regularizer: bias decay Keras Object
    :param skips: Boolean to add skip-connection between convolution blocks
    :param include_top: Boolean whether to include the fully-connected layer at the top of the network.
    :return: A Keras model instance.
    """
    if stages is None:
        stages = [1, 1, 1, 1]

    input_image = layers.Input(shape=input_shape, name=model_name)

    x = layers.BatchNormalization(axis=-1, name=model_name + '_input/bn')(input_image)
    # x = layers.LayerNormalization(axis=-1, name=model_name + '_init_bn')(input_image)

    residuals_list = []

    # stage1
    x = GConv_Block(x, gconv, ksize=3, in_channels=input_shape[-1], out_channels=out_channels,
                    h_input='Z3', h_output=h_output, kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    dropout_rate=dropout_rate, strides=(2, 2, 1), name=model_name + '_stage1')

    residuals_list.append(x)

    # stage2 and on
    for i, blocks in enumerate(stages):
        stage = i + 2

        # dropout only in odd stages
        if stage == 2 or stage == 4:
            dropout = 0
        else:
            dropout = dropout_rate

        x = layers.MaxPooling3D(pool_size=3, strides=2, padding='same',
                                name=model_name + '_stage' + str(stage) + '/pool')(x)

        stage_name = '_stage' + str(stage)

        for j in range(blocks):
            block_name = '_unit' + str(j + 1)
            name = model_name + stage_name + block_name

            if (stage == 2 or stage == 4) and (j == 0):

                # double the number of output channels at the beginning of stage 2 and 4
                x = GConv_Block(x, gconv, ksize=kernel_size, in_channels=out_channels, out_channels=out_channels*2,
                                h_input=h_output, h_output=h_output, kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                dropout_rate=dropout, name=name)
                out_channels = out_channels * 2

                # not using skip-connections when doubling the number of output channels
                # may use 1x1x1 convs to increment number of output channels of the identity part to match residuals

            else:

                # keep the same number of output channels in stage 3 and 5
                res = GConv_Block(x, gconv, ksize=kernel_size, in_channels=out_channels, out_channels=out_channels,
                                  h_input=h_output, h_output=h_output, kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                  dropout_rate=dropout, name=name)

                # add skip-connetions to groupnets
                if skips is True:
                    x = layers.add([x, res], name=name + '_add')
                else:
                    x = res

            # squeeze and excitation blocks at the end of stage 4 and 5
            if use_se is True:
                if stage == 4 or stage == 5:
                    x = SE_block(input_tensor=x, ratio=16, kernel_regularizer=kernel_regularizer,
                                 initializer=kernel_initializer, bias_regularizer=bias_regularizer, name=name + '_se')

        residuals_list.append(x)

    x = layers.GlobalAveragePooling3D(name=model_name + '_avg_pool')(x)
    # x = layers.Dropout(rate=final_dropout, name=model_name + '_final_dropout')(x)

    model = Model(input_image, x, name=model_name)

    return model, residuals_list[:-1]


if __name__ == '__main__':
    model, _ = GroupNet((192, 192, 50, 1), stages=[1, 1, 1, 1], out_channels=4, dropout_rate=0., skips=False,
                     gconv=True, use_se=False)
    model.summary(line_length=130)
