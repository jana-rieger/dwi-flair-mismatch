import logging

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import coral_ordinal as coral
from termcolor import colored

from Architectures.clinical import ClinicalNN as NN
from Architectures.groupnet import GroupNet
from Architectures.decoder import Decoder


class Empty(Exception):
    pass


class DimensionsMismatch(Exception):
    pass


def get_CNN(hyperparams, input_shape, name):
    kernel_reg = tf.keras.regularizers.l2(hyperparams['kernel_regularizer'])
    bias_reg = tf.keras.regularizers.l2(hyperparams['kernel_regularizer'])

    stages = hyperparams['stages']
    init_filters = hyperparams['init_filters']

    if 'convnet' in hyperparams['arch']:
        # baseline for fair direct comparison against GConv-Architecture
        # number of initial output feature channels gets increased by sqrt(|D4H|)=4 to roughly match the same number
        # of parameters of the GCNN. As described in http://arxiv.org/abs/1804.04656
        CNN, residuals_list = GroupNet(input_shape,
                                       name,
                                       gconv=False,
                                       stages=stages,
                                       out_channels=init_filters * 4,
                                       kernel_size=hyperparams['kernel_size'],
                                       dropout_rate=hyperparams['dropout_rate'],
                                       kernel_initializer=hyperparams['kernel_initializer'],
                                       kernel_regularizer=kernel_reg,
                                       bias_regularizer=bias_reg,
                                       skips=hyperparams['skips'],
                                       use_se=hyperparams['use_se'])

    elif 'groupnet' in hyperparams['arch']:
        CNN, residuals_list = GroupNet(input_shape,
                                       name,
                                       gconv=True,
                                       stages=stages,
                                       out_channels=init_filters,
                                       h_output=hyperparams['h_output'],
                                       kernel_size=hyperparams['kernel_size'],
                                       dropout_rate=hyperparams['dropout_rate'],
                                       kernel_initializer=hyperparams['kernel_initializer'],
                                       kernel_regularizer=kernel_reg,
                                       bias_regularizer=bias_reg,
                                       skips=hyperparams['skips'],
                                       use_se=hyperparams['use_se'])

    else:
        raise ValueError('Specify model architecture')

    return CNN, residuals_list


def embedding_block(modality, cnn_out, hyperparams, kernel_reg, bias_reg, bias_init):
    emb = layers.Dense(hyperparams['embedding_img_dim'], activation='linear',
                       kernel_regularizer=kernel_reg,
                       bias_regularizer=bias_reg,
                       bias_initializer=bias_init,
                       name=modality + '_embedding')(cnn_out)

    emb = layers.BatchNormalization(axis=-1, beta_initializer=bias_init,
                                    beta_regularizer=bias_reg,
                                    name=modality + '_embedding/bn')(emb)
    emb = layers.Activation('relu', name=modality + '_embedding/relu')(emb)

    return emb


def build_model(input_shapes, hyperparams, num_classes, gridsearch_idx=0,
                trained_dwi_model=None, trained_dwi_ae_model=None, trained_flair_ae_model=None, connect_residuals=False):
    # Build imaging streams
    # use same weight decay value for kernel and bias for simplicity
    # global tabular_model
    kernel_reg = tf.keras.regularizers.l2(hyperparams['kernel_regularizer'])
    bias_reg = tf.keras.regularizers.l2(hyperparams['kernel_regularizer'])
    bias_init = tf.keras.initializers.Constant(0.)

    inputs = []
    outputs = []

    mode = hyperparams['mode']
    if mode == 'mmop':
        mode = 'dwi_flair_clin'

    if 'dwi' in mode:
        # check if there is a pre-trained dwi model
        if trained_dwi_model is None and trained_dwi_ae_model is None:
            DWI_CNN, dwi_residuals_list = get_CNN(hyperparams, input_shapes['dwi'], 'dwi')
            dwi_input = DWI_CNN.input
            dwi_output = DWI_CNN.output

            if not connect_residuals:
                dwi_residuals_list = None

            if '_AE' in hyperparams['arch']:
                dwi_dec_out_shape = (*hyperparams['downsampling_size'], dwi_input.shape[-1])
                dwi_decoder_output = Decoder(DWI_CNN.layers[-2].output, dwi_dec_out_shape,
                                             ksize=hyperparams['kernel_size'],
                                             decoder_stages=hyperparams['decoder_stages'],
                                             kernel_initializer=hyperparams['kernel_initializer'],
                                             kernel_regularizer=hyperparams['decoder_regularizer'],
                                             dropout_rate=hyperparams['decoder_dropout_rate'],
                                             modality='dwi', from_embedding=False,
                                             residuals_list=dwi_residuals_list)

        elif trained_dwi_model is not None and trained_dwi_ae_model is None:
            print(colored('Using pretrained DWI layers.', 'cyan'))
            # The following renaming is due to unclear error thrown by h5py.
            for w in trained_dwi_model.weights:
                w._handle_name = 'PRETRAINED_' + '_' + w.name

            # Optionally freeze dwi layers.
            if hyperparams['dwi_freeze']:
                for layer in trained_dwi_model.layers:
                    layer.trainable = False
                print(colored('Freezing DWI layers.', 'yellow'))
            else:
                print(colored('Tuning DWI layers.', 'yellow'))

            dwi_input = trained_dwi_model.input
            dwi_output = trained_dwi_model.layers[-2]
        elif trained_dwi_ae_model is not None and trained_dwi_model is None:
            print(colored('Using pretrained DWI AE encoder layers.', 'cyan'))
            # The following renaming is due to unclear error thrown by h5py.
            for w in trained_dwi_ae_model.weights:
                w._handle_name = 'PRETRAINED_' + '_' + w.name

            idx_first_bottleneck_layer = None
            for i, l in enumerate(trained_dwi_ae_model.layers):
                if l.name == 'dwi_decod_bottleneck/conv':
                    idx_first_bottleneck_layer = i
            dwi_encoder_penultimate_layer = trained_dwi_ae_model.layers[idx_first_bottleneck_layer - 1]

            dwi_input = trained_dwi_ae_model.input
            dwi_output = layers.GlobalAveragePooling3D(name='dwi_avg_pool')(dwi_encoder_penultimate_layer.output)

            if '_AE' in hyperparams['arch']:
                print(colored('Using pretrained DWI AE decoder layers.', 'cyan'))
                dwi_decoder_output = trained_dwi_ae_model.output
        else:
            raise AssertionError('Cannot train model with both pretrained DWI CNN and pretrained AE.')

        dwi_embedding = embedding_block('dwi', dwi_output, hyperparams, kernel_reg, bias_reg, bias_init)

        inputs.append(dwi_input)
        outputs.append(dwi_embedding)

    if 'flair' in mode:
        # check if there is a pre-trained flair model
        if trained_flair_ae_model is None:
            FLAIR_CNN, flair_residuals_list = get_CNN(hyperparams, input_shapes['flair'], 'flair')
            flair_input = FLAIR_CNN.input
            flair_output = FLAIR_CNN.output

            if not connect_residuals:
                flair_residuals_list = None

            if '_AE' in hyperparams['arch']:
                flair_dec_out_shape = (*hyperparams['downsampling_size'], flair_input.shape[-1])
                flair_decoder_output = Decoder(FLAIR_CNN.layers[-2].output, flair_dec_out_shape,
                                             ksize=hyperparams['kernel_size'],
                                             decoder_stages=hyperparams['decoder_stages'],
                                             kernel_initializer=hyperparams['kernel_initializer'],
                                             kernel_regularizer=hyperparams['decoder_regularizer'],
                                             dropout_rate=hyperparams['decoder_dropout_rate'],
                                             modality='flair', from_embedding=False,
                                             residuals_list=flair_residuals_list)

        else:
            print(colored('Using pretrained FLAIR AE encoder layers.', 'cyan'))
            # The following renaming is due to unclear error thrown by h5py.
            for w in trained_flair_ae_model.weights:
                w._handle_name = 'PRETRAINED_' + '_' + w.name

            idx_first_bottleneck_layer = None
            for i, l in enumerate(trained_flair_ae_model.layers):
                if l.name == 'flair_decod_bottleneck/conv':
                    idx_first_bottleneck_layer = i
            flair_encoder_penultimate_layer = trained_flair_ae_model.layers[idx_first_bottleneck_layer - 1]

            flair_input = trained_flair_ae_model.input
            flair_output = layers.GlobalAveragePooling3D(name='flair_avg_pool')(flair_encoder_penultimate_layer.output)

            if '_AE' in hyperparams['arch']:
                print(colored('Using pretrained FLAIR AE decoder layers.', 'cyan'))
                flair_decoder_output = trained_flair_ae_model.output

        flair_embedding = embedding_block('flair', flair_output, hyperparams, kernel_reg, bias_reg, bias_init)

        inputs.append(flair_input)
        outputs.append(flair_embedding)

    if 'clin' in mode:
        tabular_model = NN(input_shape=input_shapes['clin'],
                           model_name='clin',
                           depths_of_layers=hyperparams['clinical_depths_of_layers'],
                           kernel_regularizer=kernel_reg,
                           bias_regularizer=bias_reg,
                           dropout_rate=hyperparams['dropout_rate'],
                           final_dropout=hyperparams['final_dropout'])

        # append clinical input
        inputs.append(tabular_model.input)
        clin_embedding = layers.Dense(hyperparams['embedding_clin_dim'], activation=tf.nn.relu,
                                      kernel_regularizer=kernel_reg,
                                      bias_regularizer=bias_reg,
                                      bias_initializer=bias_init,
                                      name='clin_embedding')(tabular_model.output)
        outputs.append(clin_embedding)

    # Concatenate embeddings.
    outputs = tf.concat(outputs, axis=-1)

    # Add final classification layers.
    for i, units in enumerate(hyperparams['depths_of_classification_layers']):
        outputs = layers.Dense(units, activation='linear',
                               kernel_regularizer=kernel_reg,
                               bias_regularizer=bias_reg,
                               bias_initializer=bias_init,
                               name='classification_dense_' + str(i))(outputs)

        outputs = layers.BatchNormalization(axis=-1, beta_initializer=bias_init,
                                            beta_regularizer=bias_reg,
                                            name='classification_dense_' + str(i) + '/bn')(outputs)
        outputs = layers.Activation('relu',
                                    name='classification_dense_' + str(i) + '/relu')(outputs)

        if hyperparams['dropout_rate'] != 0:
            outputs = layers.Dropout(rate=hyperparams['dropout_rate'],
                                     name='classification_dense_' + str(i) + '/drop')(outputs)

    # Discard the classification layer when there are no imaging models.
    if mode == 'clin':
        inputs = tabular_model.input
        outputs = tabular_model.output

    # Final layer
    if '_OR' in hyperparams['arch']:
        # for ordinal classification
        final_output = coral.CoralOrdinal(num_classes=num_classes)(outputs)
    elif '_REG' in hyperparams['arch']:
        # for regression
        final_output = layers.Dense(1,
                                    kernel_regularizer=kernel_reg,
                                    bias_regularizer=bias_reg,
                                    name='final_output_linear')(outputs)
    else:
        # for classification
        if num_classes == 2:
            final_output = layers.Dense(1, activation=tf.nn.sigmoid,
                                        kernel_regularizer=kernel_reg,
                                        bias_regularizer=bias_reg,
                                        name='final_output_sigmoid')(outputs)
        elif num_classes > 2:
            final_output = layers.Dense(num_classes, activation=tf.nn.softmax,
                                        kernel_regularizer=kernel_reg,
                                        bias_regularizer=bias_reg,
                                        name='final_output_softmax')(outputs)
        else:
            raise ValueError('Cannot define final classification layer. The number of classes is smaller then 2.'
                             'Set correct number of classes in MAIN.py script.')

    if '_AE' in hyperparams['arch']:
        if 'dwi' in mode and 'flair' in mode:
            model = Model(inputs, [final_output, dwi_decoder_output, flair_decoder_output],
                          name=mode + '_' + hyperparams['arch'])
        elif 'dwi' in mode:
            model = Model(inputs, [final_output, dwi_decoder_output],
                          name=mode + '_' + hyperparams['arch'])
        elif 'flair' in mode:
            model = Model(inputs, [final_output, flair_decoder_output],
                          name=mode + '_' + hyperparams['arch'])
        else:
            raise ValueError('Can define AE model only for modes including: "dwi" and/or "flair".')

    else:
        model = Model(inputs, final_output, name=mode + '_' + hyperparams['arch'])

    if gridsearch_idx == 0:
        model.summary(line_length=150)

    return model


def build_ae_vanilla_model(vol_input_shape, hyperparams, gridsearch_idx=0, connect_residuals=False):

    mode = hyperparams['mode']
    if mode == 'mmop':
        mode = 'dwi_flair_clin'

    if 'dwi' == mode:
        cnn_name = 'dwi'
    elif 'flair' == mode:
        cnn_name = 'flair'
    else:
        raise ValueError('Mode can be either "dwi" or "flair".')

    encoder, residuals_list = get_CNN(hyperparams, vol_input_shape, cnn_name)
    encoder_input = encoder.input
    encoder_output = encoder.layers[-2].output

    if not connect_residuals:
        residuals_list = None

    dec_out_shape = (*hyperparams['downsampling_size'], encoder_input.shape[-1])
    decoder_output = Decoder(encoder_output, dec_out_shape, ksize=hyperparams['kernel_size'],
                                             decoder_stages=hyperparams['decoder_stages'],
                                             kernel_initializer=hyperparams['kernel_initializer'],
                                             kernel_regularizer=hyperparams['decoder_regularizer'],
                                             dropout_rate=hyperparams['decoder_dropout_rate'],
                                             modality=cnn_name, from_embedding=False,
                                             residuals_list=residuals_list)

    model = Model(encoder_input, decoder_output, name=mode + '_' + hyperparams['arch'])

    if gridsearch_idx == 0:
        model.summary(line_length=150)

    return model


def build_dwi_model(dwi_input_shape, hyperparams, num_classes, gridsearch_idx=0, pretrain_mode='reg'):
    kernel_reg = tf.keras.regularizers.l2(hyperparams['kernel_regularizer'])
    bias_reg = tf.keras.regularizers.l2(hyperparams['kernel_regularizer'])

    mode = hyperparams['mode']
    if mode == 'mmop':
        mode = 'dwi_flair_clin'

    DWI_CNN = get_CNN(hyperparams, dwi_input_shape, 'dwi')

    if pretrain_mode == 'reg':
        final_output = layers.Dense(1,
                                    kernel_regularizer=kernel_reg,
                                    bias_regularizer=bias_reg,
                                    name='dwi_final_output_linear')(DWI_CNN.output)
    elif pretrain_mode == 'class':
        if num_classes == 2:
            final_output = layers.Dense(1, activation=tf.nn.sigmoid,
                                        kernel_regularizer=kernel_reg,
                                        bias_regularizer=bias_reg,
                                        name='dwi_final_output_sigmoid')(DWI_CNN.output)
        elif num_classes > 2:
            final_output = layers.Dense(num_classes, activation=tf.nn.softmax,
                                        kernel_regularizer=kernel_reg,
                                        bias_regularizer=bias_reg,
                                        name='dwi_final_output_softmax')(DWI_CNN.output)
        else:
            raise ValueError('Cannot define final classification layer in pretrain model. '
                             'The number of classes is smaller then 2.')
    else:
        raise ValueError('pretrain_mode can be set only to "reg" for regression or "class" for binary classification.')

    model = Model(DWI_CNN.input, final_output, name=mode + '_' + hyperparams['arch'])

    if gridsearch_idx == 0:
        model.summary(line_length=150)

    for w in model.weights:
        logging.debug(w.name)

    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    def groupnet_AE_regularization():
        mode = 'dwi'  # _flair
        dwi_shape = (192, 192, 48, 2)
        input_shapes = {'dwi': dwi_shape}
        decoder_stages = 4
        num_classes = 2
        arch = 'groupnet_AE'

        hyperparams = {'arch': arch,
                       'mode': mode,
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': tf.keras.losses.BinaryCrossentropy(),  # tf.keras.losses.BinaryCrossentropy()
                       'metrics': ['accuracy', 'mae', tf.keras.metrics.AUC()],  # f1_score
                       'final_assessment': ['auc', 'skl_auc', 'bacc', 'acc', 'mae', 'rmse'],
                       # 'metrics': ['mae'],
                       # 'final_assessment': ['bacc', 'acc', 'mae', 'rmse'],
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': True,
                       'use_importance_weights': False,
                       'expand_dims': [{'clin': None, 'dwi': -1, 'flair': -1}, None],
                       # 'expand_dims': [{'clin': None, 'dwi': None, 'flair': None}, None],
                       'dimensions': None,  # for strokenet
                       'stages': [1,1,1,1],
                       'version': 'v2',  # for mobilenet
                       'expansion_factor': 2,  # for mobilenet
                       'bottleneck': True,
                       'groups': 0,
                       'growth_rate': 4,  # for DenseNet
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'conv_type': 'normal',
                       'num_stages': 4,
                       'depth': 2,  # for StrokeNet
                       'init_filters': 8,
                       'dropout_rate': 0.1,
                       'final_dropout': 0,  # don't use, not consistently implemented in the framework
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'clinical_depths_of_layers': [64, 128, 64],
                       'embedding_img_dim': 256,
                       'embedding_clin_dim': 256,
                       'depths_of_classification_layers': [256],
                       'dich_class_threshold': 2,
                       'alpha': 0,  # for combined loss
                       'dwi_freeze': False,  # for pretraining
                       'decoder_stages': decoder_stages,
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48)
                       }

        build_model(input_shapes, hyperparams, num_classes, gridsearch_idx=0, trained_dwi_ae_model=None)


    def AE_vanilla_model():
        mode = 'dwi'  # _flair
        dwi_shape = (192, 192, 48, 2)
        decoder_stages = 4
        arch = 'groupnet_vanilla_AE'

        hyperparams = {'arch': arch,
                       'mode': mode,
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': tf.keras.losses.BinaryCrossentropy(),  # tf.keras.losses.BinaryCrossentropy()
                       'metrics': ['accuracy', 'mae', tf.keras.metrics.AUC()],  # f1_score
                       'final_assessment': ['auc', 'skl_auc', 'bacc', 'acc', 'mae', 'rmse'],
                       # 'metrics': ['mae'],
                       # 'final_assessment': ['bacc', 'acc', 'mae', 'rmse'],
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': False,
                       'use_importance_weights': False,
                       # 'expand_dims': [{'clin': None, 'dwi': -1, 'flair': -1}, None],
                       'expand_dims': [{'clin': None, 'dwi': None, 'flair': None}, None],
                       'dimensions': None,  # for strokenet
                       'stages': [1, 1, 1, 1],
                       'version': 'v2',  # for mobilenet
                       'expansion_factor': 2,  # for mobilenet
                       'bottleneck': True,
                       'groups': 0,
                       'growth_rate': 4,  # for DenseNet
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'conv_type': 'normal',
                       'num_stages': 4,
                       'depth': 2,  # for StrokeNet
                       'init_filters': 8,
                       'dropout_rate': 0.1,
                       'final_dropout': 0,  # don't use, not consistently implemented in the framework
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'clinical_depths_of_layers': [64, 128, 64],
                       'embedding_img_dim': 256,
                       'embedding_clin_dim': 256,
                       'depths_of_classification_layers': [256],
                       'dich_class_threshold': 2,
                       'alpha': 0,  # for combined loss
                       'dwi_freeze': False,  # for pretraining
                       'decoder_stages': decoder_stages,
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48)
                       }

        build_ae_vanilla_model(dwi_shape, hyperparams, gridsearch_idx=0)


    def AE_vanilla_model_with_residuals():
        mode = 'dwi'  # _flair
        dwi_shape = (192, 192, 48, 2)
        decoder_stages = 4
        arch = 'groupnet_pretrained_AE'

        hyperparams = {'arch': arch,
                       'mode': mode,
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': tf.keras.losses.BinaryCrossentropy(),  # tf.keras.losses.BinaryCrossentropy()
                       'metrics': ['accuracy', 'mae', tf.keras.metrics.AUC()],  # f1_score
                       'final_assessment': ['auc', 'skl_auc', 'bacc', 'acc', 'mae', 'rmse'],
                       # 'metrics': ['mae'],
                       # 'final_assessment': ['bacc', 'acc', 'mae', 'rmse'],
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': False,
                       'use_importance_weights': False,
                       # 'expand_dims': [{'clin': None, 'dwi': -1, 'flair': -1}, None],
                       'expand_dims': [{'clin': None, 'dwi': None, 'flair': None}, None],
                       'dimensions': None,  # for strokenet
                       'stages': [1, 1, 1, 1],
                       'version': 'v2',  # for mobilenet
                       'expansion_factor': 2,  # for mobilenet
                       'bottleneck': True,
                       'groups': 0,
                       'growth_rate': 4,  # for DenseNet
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'conv_type': 'normal',
                       'num_stages': 4,
                       'depth': 2,  # for StrokeNet
                       'init_filters': 8,
                       'dropout_rate': 0.1,
                       'final_dropout': 0,  # don't use, not consistently implemented in the framework
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'clinical_depths_of_layers': [64, 128, 64],
                       'embedding_img_dim': 256,
                       'embedding_clin_dim': 256,
                       'depths_of_classification_layers': [256],
                       'dich_class_threshold': 2,
                       'alpha': 0,  # for combined loss
                       'dwi_freeze': False,  # for pretraining
                       'decoder_stages': decoder_stages,
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48)
                       }

        build_ae_vanilla_model(dwi_shape, hyperparams, gridsearch_idx=0, connect_residuals=True)


    def groupnet_AE_pretrained():
        mode = 'dwi'  # _flair
        dwi_shape = (192, 192, 48, 2)
        input_shapes = {'dwi': dwi_shape}
        decoder_stages = 4
        num_classes = 2
        arch = 'groupnet_AE_pretrained'

        hyperparams = {'arch': arch,
                       'mode': mode,
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': tf.keras.losses.BinaryCrossentropy(),  # tf.keras.losses.BinaryCrossentropy()
                       'metrics': ['accuracy', 'mae', tf.keras.metrics.AUC()],  # f1_score
                       'final_assessment': ['auc', 'skl_auc', 'bacc', 'acc', 'mae', 'rmse'],
                       # 'metrics': ['mae'],
                       # 'final_assessment': ['bacc', 'acc', 'mae', 'rmse'],
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': True,
                       'use_importance_weights': False,
                       'expand_dims': [{'clin': None, 'dwi': -1, 'flair': -1}, None],
                       # 'expand_dims': [{'clin': None, 'dwi': None, 'flair': None}, None],
                       'dimensions': None,  # for strokenet
                       'stages': [1, 1, 1, 1],
                       'version': 'v2',  # for mobilenet
                       'expansion_factor': 2,  # for mobilenet
                       'bottleneck': True,
                       'groups': 0,
                       'growth_rate': 4,  # for DenseNet
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'conv_type': 'normal',
                       'num_stages': 4,
                       'depth': 2,  # for StrokeNet
                       'init_filters': 8,
                       'dropout_rate': 0.1,
                       'final_dropout': 0,  # don't use, not consistently implemented in the framework
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'clinical_depths_of_layers': [64, 128, 64],
                       'embedding_img_dim': 256,
                       'embedding_clin_dim': 256,
                       'depths_of_classification_layers': [256],
                       'dich_class_threshold': 2,
                       'alpha': 0,  # for combined loss
                       'dwi_freeze': False,  # for pretraining
                       'decoder_stages': decoder_stages,
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48)
                       }

        ae_vanilla_model = build_ae_vanilla_model(dwi_shape, hyperparams, gridsearch_idx=0)

        build_model(input_shapes, hyperparams, num_classes, gridsearch_idx=0, trained_dwi_model=None,
                    trained_dwi_ae_model=ae_vanilla_model)


    def groupnet_dwi_flair_pretrained():
        mode = 'dwi_flair'  # _flair
        dwi_shape = (192, 192, 48, 1)
        flair_shape = (192, 192, 48, 1)
        input_shapes = {'dwi': dwi_shape,
                        'flair': flair_shape}
        decoder_stages = 4
        num_classes = 2
        arch = 'groupnet_pretrained'

        hyperparams = {'arch': arch,
                       'mode': mode,
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': tf.keras.losses.BinaryCrossentropy(),  # tf.keras.losses.BinaryCrossentropy()
                       'metrics': ['accuracy', 'mae', tf.keras.metrics.AUC()],  # f1_score
                       'final_assessment': ['auc', 'skl_auc', 'bacc', 'acc', 'mae', 'rmse'],
                       # 'metrics': ['mae'],
                       # 'final_assessment': ['bacc', 'acc', 'mae', 'rmse'],
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': True,
                       'use_importance_weights': False,
                       'expand_dims': [{'clin': None, 'dwi': -1, 'flair': -1}, None],
                       # 'expand_dims': [{'clin': None, 'dwi': None, 'flair': None}, None],
                       'dimensions': None,  # for strokenet
                       'stages': [1,1,1,1],
                       'version': 'v2',  # for mobilenet
                       'expansion_factor': 2,  # for mobilenet
                       'bottleneck': True,
                       'groups': 0,
                       'growth_rate': 4,  # for DenseNet
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'conv_type': 'normal',
                       'num_stages': 4,
                       'depth': 2,  # for StrokeNet
                       'init_filters': 8,
                       'dropout_rate': 0.1,
                       'final_dropout': 0,  # don't use, not consistently implemented in the framework
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'clinical_depths_of_layers': [64, 128, 64],
                       'embedding_img_dim': 256,
                       'embedding_clin_dim': 256,
                       'depths_of_classification_layers': [256],
                       'dich_class_threshold': 2,
                       'alpha': 0,  # for combined loss
                       'dwi_freeze': False,  # for pretraining
                       'decoder_stages': decoder_stages,
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48)
                       }

        dwi_hyperparams = {k: v for k, v in hyperparams.items()}
        dwi_hyperparams['mode'] = 'dwi'
        flair_hyperparams = {k: v for k, v in hyperparams.items()}
        flair_hyperparams['mode'] = 'flair'

        dwi_ae_vanilla_model = build_ae_vanilla_model(dwi_shape, dwi_hyperparams, gridsearch_idx=0)
        flair_ae_vanilla_model = build_ae_vanilla_model(flair_shape, flair_hyperparams, gridsearch_idx=0)

        build_model(input_shapes, hyperparams, num_classes, gridsearch_idx=0, trained_dwi_model=None,
                    trained_dwi_ae_model=dwi_ae_vanilla_model, trained_flair_ae_model=flair_ae_vanilla_model)


    def groupnet_dwi_pretrained():
        mode = 'dwi'  # _flair
        dwi_shape = (192, 192, 48, 1)
        flair_shape = (192, 192, 48, 1)
        input_shapes = {'dwi': dwi_shape,
                        'flair': flair_shape}
        decoder_stages = 4
        num_classes = 2
        arch = 'groupnet_pretrained'

        hyperparams = {'arch': arch,
                       'mode': mode,
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': tf.keras.losses.BinaryCrossentropy(),  # tf.keras.losses.BinaryCrossentropy()
                       'metrics': ['accuracy', 'mae', tf.keras.metrics.AUC()],  # f1_score
                       'final_assessment': ['auc', 'skl_auc', 'bacc', 'acc', 'mae', 'rmse'],
                       # 'metrics': ['mae'],
                       # 'final_assessment': ['bacc', 'acc', 'mae', 'rmse'],
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': True,
                       'use_importance_weights': False,
                       'expand_dims': [{'clin': None, 'dwi': -1, 'flair': -1}, None],
                       # 'expand_dims': [{'clin': None, 'dwi': None, 'flair': None}, None],
                       'dimensions': None,  # for strokenet
                       'stages': [1,1,1,1],
                       'version': 'v2',  # for mobilenet
                       'expansion_factor': 2,  # for mobilenet
                       'bottleneck': True,
                       'groups': 0,
                       'growth_rate': 4,  # for DenseNet
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'conv_type': 'normal',
                       'num_stages': 4,
                       'depth': 2,  # for StrokeNet
                       'init_filters': 8,
                       'dropout_rate': 0.1,
                       'final_dropout': 0,  # don't use, not consistently implemented in the framework
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'clinical_depths_of_layers': [64, 128, 64],
                       'embedding_img_dim': 256,
                       'embedding_clin_dim': 256,
                       'depths_of_classification_layers': [256],
                       'dich_class_threshold': 2,
                       'alpha': 0,  # for combined loss
                       'dwi_freeze': False,  # for pretraining
                       'decoder_stages': decoder_stages,
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48)
                       }

        dwi_hyperparams = {k: v for k, v in hyperparams.items()}
        dwi_hyperparams['mode'] = 'dwi'

        dwi_ae_vanilla_model = build_ae_vanilla_model(dwi_shape, dwi_hyperparams, gridsearch_idx=0)

        build_model(input_shapes, hyperparams, num_classes, gridsearch_idx=0, trained_dwi_model=None,
                    trained_dwi_ae_model=dwi_ae_vanilla_model, trained_flair_ae_model=None)


    def groupnet_flair_pretrained():
        mode = 'flair'  # _flair
        dwi_shape = (192, 192, 48, 1)
        flair_shape = (192, 192, 48, 1)
        input_shapes = {'dwi': dwi_shape,
                        'flair': flair_shape}
        decoder_stages = 4
        num_classes = 2
        arch = 'groupnet_pretrained'

        hyperparams = {'arch': arch,
                       'mode': mode,
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': tf.keras.losses.BinaryCrossentropy(),  # tf.keras.losses.BinaryCrossentropy()
                       'metrics': ['accuracy', 'mae', tf.keras.metrics.AUC()],  # f1_score
                       'final_assessment': ['auc', 'skl_auc', 'bacc', 'acc', 'mae', 'rmse'],
                       # 'metrics': ['mae'],
                       # 'final_assessment': ['bacc', 'acc', 'mae', 'rmse'],
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': True,
                       'use_importance_weights': False,
                       'expand_dims': [{'clin': None, 'dwi': -1, 'flair': -1}, None],
                       # 'expand_dims': [{'clin': None, 'dwi': None, 'flair': None}, None],
                       'dimensions': None,  # for strokenet
                       'stages': [1,1,1,1],
                       'version': 'v2',  # for mobilenet
                       'expansion_factor': 2,  # for mobilenet
                       'bottleneck': True,
                       'groups': 0,
                       'growth_rate': 4,  # for DenseNet
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'conv_type': 'normal',
                       'num_stages': 4,
                       'depth': 2,  # for StrokeNet
                       'init_filters': 8,
                       'dropout_rate': 0.1,
                       'final_dropout': 0,  # don't use, not consistently implemented in the framework
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'clinical_depths_of_layers': [64, 128, 64],
                       'embedding_img_dim': 256,
                       'embedding_clin_dim': 256,
                       'depths_of_classification_layers': [256],
                       'dich_class_threshold': 2,
                       'alpha': 0,  # for combined loss
                       'dwi_freeze': False,  # for pretraining
                       'decoder_stages': decoder_stages,
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48)
                       }

        flair_hyperparams = {k: v for k, v in hyperparams.items()}
        flair_hyperparams['mode'] = 'flair'

        flair_ae_vanilla_model = build_ae_vanilla_model(flair_shape, flair_hyperparams, gridsearch_idx=0)

        build_model(input_shapes, hyperparams, num_classes, gridsearch_idx=0, trained_dwi_model=None,
                    trained_dwi_ae_model=None, trained_flair_ae_model=flair_ae_vanilla_model)



    # AE_vanilla_model()
    # AE_vanilla_model_with_residuals()
    # groupnet_AE_pretrained()
    # groupnet_dwi_flair_pretrained()
    # groupnet_dwi_pretrained()
    # groupnet_flair_pretrained()
    groupnet_AE_regularization()
