import copy
import csv
import glob
import logging
import os
import random
import shutil
from os.path import join

import coral_ordinal as coral
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from termcolor import colored

from ADT.DataGeneratorADT import DataGenerator
from ADT.ModelADT import DimensionsMismatch
from ADT.ModelADT import build_model, build_dwi_model, build_ae_vanilla_model
from ADT.helpers import make_folder, plot_loss_acc_history
from ADT.losses import ae_loss
from ADT.metrics import sensivity_specifity_cutoff, sensivity_calibrated_cutoff, specifity_calibrated_cutoff
from Architectures.GroupConv3D import GroupConv3D


class Empty(Exception):
    pass


class Pipeline:

    def __init__(self, config_dict):
        self._config = config_dict
        self._architecture = None
        self._default_params = None
        self._model = None
        self._validation = {}
        self.dwi_mean = 0
        self.dwi_std = 1
        self.flair_mean = 0
        self.flair_std = 1
        self.num_data_indices = []

    def preprocess_specific_data(self, batch, key):
        """
        Preprocess data: center numeric data, and standardize imaging

        The function is only applicable on np.ndarray types.
        """
        processed_batch = copy.deepcopy(batch)

        if key == "clin":
            for i, ind in enumerate(self.num_data_indices):
                processed_batch[:, ind] = (batch[:, ind] - self.clin_mean[i]) / self.clin_std[i]
        elif key == "dwi":
            processed_batch = (batch - self.dwi_mean) / self.dwi_std
        elif key == "flair":
            processed_batch = (batch - self.flair_mean) / self.flair_std
        else:
            print("Key: " + str(key) + " not supported for preprocessing.")

        return processed_batch

    def preprocess_all_data(self, batch, keys):
        """
        Preprocess data: center numeric data, and standardize imaging

        The function is only applicable on np.ndarray types.
        """

        for key in keys:
            batch[key] = self.preprocess_specific_data(batch[key], key)

        return batch

    def createDG(self, filenameRegex, path, ID_cut=[0, -1], ftype='csv'):
        """
        Create DataGenerator instance.
        """
        print(filenameRegex)
        DG = DataGenerator()
        DG.setPath(path)
        DG.loadDUF(filenameRegex, path, ID_cut, ftype)
        return DG

    def _length(self, DG_list):
        """
        returns the number of datasets in DG_list

        :param DG_list: a list of DataGenerator objects or dictionaries
        :return: no. of datasets in the first DataGenerator object in DG_list
        """

        if isinstance(DG_list[0], DataGenerator):
            return len(DG_list[0])
        else:
            return len(list(DG_list[0].values())[0])

    def _test_compt(self, DG_list, expand_dims):
        """
        test compatibility of DG_list and expand_dims values and dimensionality

        Make sure that DGs is a list of DataGenerator objects or dictionaries and expand_dims is None
        or has corresponding structure to DG_list

        :param DG_list: list of DataGenerator objects or dictionaries
            Exemplary input 1: DG_list = [DG_01, DG_02, DG_03]
            Exemplary input 2: [{"01": DG_01, "02": DG_02}, DG_03]
        :param expand_dims: a list of integers or dictionaries indicating which axis of the corresponding DG in DG_list
                should be expanded, or provides a None value if no expansion is necessary.
                None value indicates no dimension extension for any of the generated batches.
        :return: True if compatible, raises a specified error if not
        """

        if not isinstance(DG_list, list):
            raise TypeError('DG_list must be a list of DataGenerator objects. DG_list is not a list!')
        for i, DG in enumerate(DG_list):
            if isinstance(DG, dict):
                for key in DG.keys():
                    if not (isinstance(DG[key], DataGenerator)):
                        raise TypeError('DG_list must be a list of DataGenerator objects or dictionaries. '
                                        'One or more of the dictionaries values is not a DataGenerator object.')
                    if (expand_dims is not None) and not (key in expand_dims[i]):
                        raise TypeError('expand_dims must correspond to the structure of DG_list'
                                        'with values of int (i.e. dimension to expand) or None'
                                        '(i.e. not expant dimensions)')
            elif isinstance(DG, DataGenerator):
                if (expand_dims is not None) and (expand_dims[i] is not None) and not (isinstance(expand_dims[i], int)):
                    raise TypeError('expand_dims must correspond to the structure of DG_list with values of int '
                                    '(i.e. dimension to expand) or None (i.e. not expand any dimensions)')
            else:
                raise TypeError('DG_list must be a list of DataGenerator objects or dictionaries.')
        return True

    def getBatch(self, DG_list, IDs, output_format='np.ndarray', expand_dims=None):
        """
        Create corresponding batched data of the given IDs for each of the DataGenerators in the list DG_list

        Raise error if the lne(IDs) is larger than the stored observations in any of the given DGs
        :param DG_list: list of DataGenerator objects or dictionaries
                Examplary input 1: DG_list = [DG_01, DG_02, DG_03]
                Examplary input 2: [{"01": DG_01, "02": DG_02}, DG_03]
        :param IDs: list of IDs to batch on
        :param output_format: output format. Supported options are 'np.ndarray' (default) ; 'pd.df'.
        :param expand_dims: a list of integers or dictionaries indicating which axis of the corresponding DG in DG_list
                should be expanded, or provides a None value if no expension is necessary.
                The default value None indicates no dimension extension for any of the generated batches.
        :return: list of batches (or dictionaries of batches, corresponding to the input format) of length len(DG_list),
                each batch in the returned list has a 1st dimension == len(IDs)
        """

        # check input:
        self._test_compt(DG_list, expand_dims)
        # Make sure that batch_size is valid
        if len(IDs) < 1:
            raise ValueError('IDs must contain at least one value')
        if (isinstance(DG_list[0], DataGenerator) and len(IDs) > len(DG_list[0])) or (
                isinstance(DG_list[0], dict) and len(IDs) > len(list(DG_list[0].values())[0])):
            raise ValueError('IDs contains more IDs than the listed IDs in the DataGenerator objects')
        # Make sure that output_format is supported
        if output_format not in ['np.ndarray', 'pd.df']:
            raise ValueError('The chosen output_format is not supported. '
                             'Please use one of the supported formats \'np.ndarray\' or\'pd.df\'')

        batches = []
        for i, DG in enumerate(DG_list):
            if isinstance(DG, DataGenerator):
                batch = DG.generate(IDs, output_format)

                if expand_dims and expand_dims[i] is not None:
                    batch = np.expand_dims(batch, expand_dims[i])
                batches.append(batch)
            else:  # in case it's a dictionary of DGs, append batches with a corresponding dictionary
                DG_dict = {}
                for key, sub_DG in DG.items():
                    batch = sub_DG.generate(IDs, output_format)

                    if expand_dims and expand_dims[i][key] is not None:
                        batch = np.expand_dims(batch, expand_dims[i][key])
                    DG_dict[key] = batch
                batches.append(DG_dict)

        if len(batches) > 1:
            return tuple(batches)
        return batches[0]

    @staticmethod
    def _ae_out_dict(x, y, model_outputs, get_labels=True):
        """
        Creates a dictionary with labels for each output for the auto-encoder architecture.
        """
        out_dict = {}

        for out in model_outputs:
            if 'final_output' in out.name:
                out_dict[out.name.split('/')[0]] = y  # gets the output layer name
            elif 'decoder_output' in out.name:
                out_dict[out.name.split('/')[0]] = x[out.name.split('_')[0]]  # gets the modality
            else:
                raise NameError('Unknown model output: ' + out.name + '. Cannot match labels.')

        return out_dict

    def batchGenerator(self, DG_list, batch_size, output_format='np.ndarray', epochs=1, expand_dims=None):
        """
        Generates (randomly) batched data of the given batch size for each of the DataGenerators in the list DG_list

        The number of generated batches equals len(DG_list[0])//batch_size.
        i.e. if len(DG_list[0])%batch_size != 0, the leftover observations will not be used in any of the batches.
        Raise error if the batch_size is larger than the stored observations in any of the given DGs

        :param DG_list: list of DataGenerator objects or dictionaries
                Exemplary input 1: DG_list = [DG_01, DG_02, DG_03]
                Exemplary input 2: [{"01": DG_01, "02": DG_02}, DG_03]
        :param batch_size: integer indicating the batch size
        :param output_format: output format. Supported options are 'np.ndarray' (default) ; 'pd.df'.
        :param epochs: positive integer indicating the number of epochs. The generator will generate
                epochs*len(DG_list[0])/batch_size batches.
        :param expand_dims: a list of integers or dictionaries indicating which axis of the corresponding DG in DG_list
                should be expanded, or provides a None value if no expension is necessary.
                The default value None indicates no dimension extension for any of the generated batches.
        :return: tuple of batches (or dictionaries of batches, corresponding to the input format) of length len(DG_list),
                each batch in the returned list has a 1st dimension == batch_size
        """

        # check input:
        self._test_compt(DG_list, expand_dims)
        # Make sure that batch_size is valid
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be a positive integer')
        if batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if batch_size > self._length(DG_list):
            raise ValueError('batch_size must be smaller than the no. of datasets in the DG_list')
        # Make sure that output_format is supported
        if output_format not in ['np.ndarray', 'pd.df']:
            raise ValueError(
                'The chosen output_format is not supported. Please use one of the supported formats \'np.ndarray\' or\'pd.df\'')
        # Make sure the epochs has a valid value
        if not isinstance(epochs, int):
            raise TypeError('epochs must be an integer')
        elif epochs < 0:
            raise ValueError('epochs must be a positive integer')

        for ep in range(epochs):
            if isinstance(DG_list[0], DataGenerator):
                IDs = DG_list[0].getIDs()
            else:  # in case it's a dictionary of DGs:
                IDs = list(DG_list[0].values())[0].getIDs()
            random.shuffle(IDs)
            while len(IDs) >= batch_size:
                # select batch IDs
                batchIDs, IDs = IDs[:batch_size], IDs[batch_size:]
                batches = []
                for i, DG in enumerate(DG_list):
                    if isinstance(DG, DataGenerator):
                        # retrieve batch
                        batch = DG.generate(batchIDs, output_format)

                        # expand dims
                        if expand_dims and expand_dims[i] is not None:
                            batch = np.expand_dims(batch, expand_dims[i])

                        batches.append(copy.deepcopy(batch))

                    else:  # in case it's a dictionary of DGs, append batches with a corresponding dictionary
                        DG_dict = {}
                        for key, sub_DG in DG.items():
                            batch = sub_DG.generate(batchIDs, output_format)

                            if expand_dims and expand_dims[i][key] is not None:
                                batch = np.expand_dims(batch, expand_dims[i][key])

                            # preprocess
                            DG_dict[key] = self.preprocess_specific_data(batch, key)

                        batches.append(DG_dict)
                yield tuple(batches)  # match output type to keras fit_generator generator function

    @staticmethod
    def _downsample_volume(data_X, mode, downsampling_size):
        downsampled_data_X = {}

        for k, v in data_X.items():
            if k != 'clin' and k in mode:
                if v.shape[3] == 50:
                    # cut first and last slice from z dimension
                    downsampled_data_X[k] = v[:, :, :, 1:-1, :]
                    # downsample x an y dimensions
                    downsampled_data_X[k] = downsampled_data_X[k][:, ::2, ::2, :, :]
                else:
                    downsampled_data_X[k] = v[:, ::2, ::2, :, :]
                # downsampled_data_X[k] = resize(v, (v.shape[0], *downsampling_size, v.shape[-1]))  # too slow

        return downsampled_data_X

    def batchGenerator_AE(self, DG_X, DG_y, batch_size, downsampling_size=(1, 1, 1), mode=None,
                          output_format='np.ndarray', epochs=1, expand_dims=None,
                          models_outputs=None):
        """
        Generates (randomly) batched data of the given batch size.

        The number of generated batches equals len(DG_list[0])//batch_size.
        i.e. if len(DG_list[0])%batch_size != 0, the leftover observations will not be used in any of the batches.
        Raise error if the batch_size is larger than the stored observations in any of the given DGs

        :param DG_X: DataGenerator with training data. Dictionary with DataGenerator for each modality.
                E.g. {"clin": DG_clin, "dwi": DG_dwi}
        :param DG_y: DataGenerator with label data.
        :param batch_size: integer indicating the batch size
        :param downsampling_size:
        :param output_format: output format. Supported options are 'np.ndarray' (default) ; 'pd.df'.
        :param epochs: positive integer indicating the number of epochs. The generator will generate
                epochs*len(DG_list[0])/batch_size batches.
        :param expand_dims: a list of integers or dictionaries indicating which axis of the corresponding DG in DG_list
                should be expanded, or provides a None value if no expension is necessary.
                The default value None indicates no dimension extension for any of the generated batches.
        :param models_outputs: Output layers of the model.
        :return: tuple of batches (or dictionaries of batches, corresponding to the input format) of length len(DG_list),
                each batch in the returned list has a 1st dimension == batch_size
        """

        # check input:
        self._test_compt([DG_X, DG_y], expand_dims)
        # Make sure that batch_size is valid
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be a positive integer')
        if batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if batch_size > self._length([DG_X, DG_y]):
            raise ValueError('batch_size must be smaller than the no. of datasets in the DG_list')
        # Make sure that output_format is supported
        if output_format not in ['np.ndarray', 'pd.df']:
            raise ValueError(
                'The chosen output_format is not supported. Please use one of the supported formats \'np.ndarray\' or\'pd.df\'')
        # Make sure the epochs has a valid value
        if not isinstance(epochs, int):
            raise TypeError('epochs must be an integer')
        elif epochs < 0:
            raise ValueError('epochs must be a positive integer')

        for i in range(epochs):
            IDs = DG_y.getIDs()
            random.shuffle(IDs)
            counter = 0
            while len(IDs) >= batch_size:
                # select batch IDs
                batchIDs, IDs = IDs[:batch_size], IDs[batch_size:]

                # feature data
                data_X = {}
                for key, sub_DG in DG_X.items():
                    # retrieve sub-batch data
                    sub_batch = sub_DG.generate(batchIDs, output_format)

                    if expand_dims and expand_dims[0][key] is not None:
                        sub_batch = np.expand_dims(sub_batch, expand_dims[0][key])

                    # preprocess
                    data_X[key] = self.preprocess_specific_data(sub_batch, key)

                # label data
                # retrieve batch data
                data_y = DG_y.generate(batchIDs, output_format)

                if expand_dims and expand_dims[1] is not None:
                    data_y = np.expand_dims(data_y, expand_dims[1])

                # downsample volume label data
                downsampled_data_X = self._downsample_volume(data_X, mode, downsampling_size)

                # create batch (X, y)
                batch = data_X, self._ae_out_dict(downsampled_data_X, data_y, model_outputs=models_outputs)

                if counter == 0:
                    for k, v in batch[0].items():
                        logging.debug('train batch X %s shape: %s', k, v.shape)
                        if len(v[0]) == 0:
                            logging.debug('train batch X %s: %s', k, v)
                        else:
                            logging.debug('train batch X %s: %s', k, v[0][..., 0][0])
                    for k, v in batch[1].items():
                        logging.debug('train batch y %s shape: %s', k, v.shape)
                        if v.shape == (batch_size,):
                            logging.debug('train batch y %s: %s', k, v)
                        else:
                            logging.debug('train batch y %s: %s', k, v[0][..., 0][0])
                counter += 1

                yield batch

    def batchGenerator_AE_vanilla(self, DG_X, batch_size, downsampling_size=(1, 1, 1), mode=None,
                                  output_format='np.ndarray', epochs=1, expand_dims=None):
        """
        Generates (randomly) batched data of the given batch size.

        The number of generated batches equals len(DG_list[0])//batch_size.
        i.e. if len(DG_list[0])%batch_size != 0, the leftover observations will not be used in any of the batches.
        Raise error if the batch_size is larger than the stored observations in any of the given DGs

        :param DG_X: DataGenerator with training data. Dictionary with DataGenerator for each modality.
                E.g. {"clin": DG_clin, "dwi": DG_dwi}
        :param batch_size: integer indicating the batch size
        :param downsampling_size:
        :param output_format: output format. Supported options are 'np.ndarray' (default) ; 'pd.df'.
        :param epochs: positive integer indicating the number of epochs. The generator will generate
                epochs*len(DG_list[0])/batch_size batches.
        :param expand_dims: a list of integers or dictionaries indicating which axis of the corresponding DG in DG_list
                should be expanded, or provides a None value if no expension is necessary.
                The default value None indicates no dimension extension for any of the generated batches.
        :return: tuple of batches (or dictionaries of batches, corresponding to the input format) of length len(DG_list),
                each batch in the returned list has a 1st dimension == batch_size
        """

        # Make sure that batch_size is valid
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be a positive integer')
        if batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if batch_size > self._length([DG_X]):
            raise ValueError('batch_size must be smaller than the no. of datasets in the DG_list')
        # Make sure that output_format is supported
        if output_format not in ['np.ndarray', 'pd.df']:
            raise ValueError('The chosen output_format is not supported. '
                             'Please use one of the supported formats \'np.ndarray\' or\'pd.df\'')
        # Make sure the epochs has a valid value
        if not isinstance(epochs, int):
            raise TypeError('epochs must be an integer')
        elif epochs < 0:
            raise ValueError('epochs must be a positive integer')

        for i in range(epochs):
            IDs = list(DG_X.values())[0].getIDs()
            random.shuffle(IDs)
            counter = 0
            while len(IDs) >= batch_size:
                # select batch IDs
                batchIDs, IDs = IDs[:batch_size], IDs[batch_size:]

                # feature data
                data_X = {}
                for key, sub_DG in DG_X.items():
                    # retrieve sub-batch data
                    sub_batch = sub_DG.generate(batchIDs, output_format)

                    if expand_dims and expand_dims[0][key] is not None:
                        sub_batch = np.expand_dims(sub_batch, expand_dims[0][key])

                    # preprocess
                    data_X[key] = self.preprocess_specific_data(sub_batch, key)

                # downsample volume label data
                downsampled_data_X = self._downsample_volume(data_X, mode, downsampling_size)

                # create batch (X, y)
                batch = data_X, downsampled_data_X[mode]

                if counter == 0:
                    for k, v in batch[0].items():
                        logging.debug('train batch X %s shape: %s', k, v.shape)
                        if len(v[0]) == 0:
                            logging.debug('train batch X %s: %s', k, v)
                        else:
                            logging.debug('train batch X %s: %s', k, v[0][..., 0][0])
                    for v in batch[1]:
                        logging.debug('train batch y shape: %s', v.shape)
                        if v.shape == (batch_size,):
                            logging.debug('train batch y: %s', v)
                        else:
                            logging.debug('train batch y: %s', v[0][..., 0][0])
                counter += 1

                yield batch

    def setParams(self, params):
        """
        Set hyperparameters using the params dictionary

        Raise an error any of the parameters is not included in the default parameters
        :param params: dictionary of hyperparameters (key = name, value = value)
        :return: True when done
        """

        if not self._default_params:
            self._default_params = params
        else:
            for key in params.keys():
                if key not in self._default_params:
                    raise ValueError('The hyperparameter ', key,
                                     ' is not included as a default parameter. Use the setArchitecture method with an '
                                     'updated file or adjust your params dictionary.')
            self._default_params.update(params)
        return True

    def setStandartization(self, DG_X_dwi_train, DG_X_flair_train, DG_X_clin_train, numerical_data,
                           numerical_data_indices):

        # Get training sets

        # Imaging
        if DG_X_dwi_train is not None:
            X_dwi_train = self.getBatch([DG_X_dwi_train], DG_X_dwi_train.getIDs())
            self.dwi_mean, self.dwi_std = X_dwi_train.flatten().mean(), X_dwi_train.flatten().std()
            print('dwi_mean and std: ', self.dwi_mean, self.dwi_std)

        if DG_X_flair_train is not None:
            X_flair_train = self.getBatch([DG_X_flair_train], DG_X_flair_train.getIDs())
            self.flair_mean, self.flair_std = X_flair_train.flatten().mean(), X_flair_train.flatten().std()
            print('flair_mean and std:', self.flair_mean, self.flair_std)

        # Clinical
        print('numerical data:', numerical_data)
        print('numerical data indices:', numerical_data_indices)
        self.num_data_indices = numerical_data_indices

        if len(numerical_data) == 0 and len(numerical_data_indices) == 0:
            return True

        X_clin_train = self.getBatch([DG_X_clin_train], DG_X_clin_train.getIDs(), 'pd.df')
        X_clin_train = X_clin_train.apply(lambda x: x.astype('float64'))
        scaler = preprocessing.StandardScaler().fit(X_clin_train.loc[:, numerical_data])
        self.clin_mean, self.clin_std = scaler.mean_, scaler.scale_

        return True

    def get_class_weights(self, y_train):
        print('')
        print('Calculating class weights...')

        total = len(y_train)
        unique, counts = np.unique(y_train, return_counts=True)

        print('Training distribution:')
        for u, c in zip(unique, counts):
            print('label', u, ':', c / total)

        class_weights = {}
        for u, c in zip(unique, counts):
            class_weights[u] = total / (len(unique) * c)
            print('weight for class', u, ':', class_weights[u])

        max_unique = max(unique)
        for i in range(int(max_unique)):
            if i not in class_weights.keys():
                class_weights[i] = 100

        return class_weights

    def get_importance_weights(self, y_train):
        print('')
        print('Calculating importance weights...')

        total = len(y_train)
        unique, counts = np.unique(y_train, return_counts=True)

        print('Training distribution:')
        for u, c in zip(unique, counts):
            print('label', u, ':', c / total)

        importance_weights = np.zeros(len(unique) - 1)
        label_array = np.array(y_train)

        for i, t in enumerate(range(int(min(unique)), int(max(unique)))):
            # from: https://github.com/Raschka-research-group/coral-cnn/blob/726e54579db008d9c16868fa76b2292b9dec9fbc/experiment-logs/afad/afad-coral.py#L136
            # importance_weights[i] = np.sqrt(float(total - label_array[label_array > t].shape[0]))
            # importance_weights = importance_weights / max(importance_weights)

            importance_weights[i] = total / ((len(unique)) * label_array[label_array > t].shape[0])
            print('importance for level', i, ':', importance_weights[i])

        return importance_weights

    def train(self, DG_X_train, DG_y_train, X_val, y_val, batch_size, model, class_weights,
              importance_weights,
              epochs=1, csv_logfile='train_log.csv', tensorboard_name='tensorboard',
              model_filename='temp_model_{epoch}.h5', models_path='', hyperparams=None, lrate=None,
              plot_filename='', foldname='', opt=None, compile_model=True):
        """
        Train (and save) the model.

        :param models_path: String path to models folder
        :param tensorboard_name: name of folder to save tensorboard logs
        :param DG_X_train: DataGenerator object or dictionary of DG objects with training data
        :param DG_y_train: DataGenerator object or dictionary of DG objects with training labels
        :param X_val: Preprocessed validation data. (E.g. standardized) To make the training faster. Applying
                    standardization take considerable amount of time. -> Do this before grid search loop.
        :param y_val: Preprocessed validation labels. (E.g. standardized) To make the training faster. Applying
                    standardization take considerable amount of time.  -> Do this before grid search loop.
        :param batch_size: integer indicating batch size, must be < len(train_X)
        :param model: compiled model - notice that this is an optional input for when setArchitecture is integrated to
            Pipeline
        :param epochs: number of epochs
        :param csv_logfile: csv filename indicating where to save the training log
        :param model_filename: path+name of where to save the trained model. If None, the model is not saved.
        :param hyperparams: an optional dictionary of hyperparameters values. If none or partial list is given,
            the model is trained using the stored self._param hyperparameters.
        :param lrate: LearningRateSchedule object.
        :param plot_filename: String: File name for saving plots.
        :param foldname: String identifier of the fold.
        :param opt: Optimizer object.
        :param class_weights: Weights to tackle class imbalance in the training set.
        :param importance_weights: Weights if level importance for coral loss function.
        :param compile_model: True for compiling the model automatically by loading. False otherwise.
        :return: A tuple with the trained model, history and training duration (duration_train)
        """

        import time
        from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

        # SANITY CHECKS
        # check input:
        if not model:
            if not self._architecture:
                raise Empty('The model architecture and hyperparameters must be set before training. '
                            'Use the setArchitecture method. Otherwise set the compiled model as an input'
                            'using the model argument')
        # make sure filename is a string
        if model_filename:
            if not isinstance(model_filename, str):
                raise TypeError('filename must be a string containing the path+filename of where to save the model')
        # Check DataGenerator inputs and expand_dims compatibility:
        if DG_y_train is not None:
            self._test_compt([DG_X_train, DG_y_train], self._default_params['expand_dims'])
        # make sure that the hyperparameter (keys) correspond with the self._params
        if hyperparams:
            if not isinstance(hyperparams, dict):
                raise TypeError('hyperparameters must be a dictionary of hyperparameters names / values')
            for key in hyperparams.keys():
                if key not in self._default_params:
                    raise ValueError('The parameter ', key, ' is not defined for the model. \
                        To add this parameter, you must change the default parameters with one of the following options: \n \
                        1) Update the default hyperparameters using setParams method. \n \
                        2) Change getModel function input in the architecture file and then reload the model with the default \
                        hyperparameters using setArchitecture method. This is not an option for the current patched Pipeline version.')
        # make sure epochs is a positive integer
        if not isinstance(epochs, int):
            raise TypeError('The epochs parameter must be a positive integer')
        elif epochs < 1:
            raise ValueError('The epochs parameter must be a positive integer')
        # make sure batch_size is a positive integer smaller than the number of training sets
        if not isinstance(batch_size, int):
            raise TypeError('The batch_size parameter must be a positive integer')
        elif batch_size < 1:
            raise ValueError('The batch_size parameter must be a positive integer')
        if batch_size > self._length([DG_X_train]):
            raise ValueError('The batch_size = ', batch_size, ' value must be smaller or equal to the number'
                                                              'of training sets.')
        # make sure that X and y dimensions match (test supporting only DG inputs at the moment):
        if DG_y_train is not None:
            if self._length([DG_y_train]) != self._length([DG_X_train]):
                raise DimensionsMismatch(
                    'The number of observations in DG_X_train = ' + str(self._length([DG_X_train])) +
                    ' must equal the number of observations in DG_y_train = ' +
                    str(self._length([DG_y_train])))

        # CALLBACKS
        csv_logger = CSVLogger(csv_logfile)
        # early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3)
        temp_dir_name = 'saves_freeze' if hyperparams['dwi_freeze'] else 'saves'
        temp_models_dir = make_folder(join(models_path, temp_dir_name, foldname))
        saves_name = join(temp_models_dir, 'temp_model_{epoch}.h5')
        if '_AE' in hyperparams['arch'] and '_vanilla_AE' not in hyperparams['arch']:
            monitor = None
            for out in model.outputs:
                if 'final_output' in out.name:
                    monitor = 'val_' + out.name.split('/')[0] + '_loss'
        else:
            monitor = 'val_loss'
        if monitor is None:
            raise NameError('No final output in model outputs: ' + model.outputs + '. Cannot match monitor for '
                                                                                   'ModelCheckpoint.')
        print(colored('Monitor: ' + monitor, 'yellow'))
        model_checkpoint = ModelCheckpoint(saves_name, monitor=monitor, save_best_only=False,
                                           save_weights_only=False, mode='min')
        tensorboard_callback = TensorBoard(log_dir=tensorboard_name, histogram_freq=1)
        # Add Learning Rate Scheduler to callbacks
        if lrate is None:
            if hyperparams['tensorboard']:
                callbacks_list = [csv_logger, model_checkpoint, tensorboard_callback]
            else:
                callbacks_list = [csv_logger, model_checkpoint]
        else:
            callbacks_list = [csv_logger, model_checkpoint, tensorboard_callback, lrate]

        # TRAIN DATA
        # create training data generator, tuple the validation data and train the model:
        if '_vanilla_AE' in hyperparams['arch']:
            generator = self.batchGenerator_AE_vanilla(DG_X=DG_X_train, batch_size=batch_size,
                                                       downsampling_size=hyperparams['downsampling_size'],
                                                       mode=hyperparams['mode'], output_format='np.ndarray',
                                                       expand_dims=hyperparams['expand_dims'], epochs=epochs)
        elif '_AE' in hyperparams['arch']:
            generator = self.batchGenerator_AE(DG_X=DG_X_train, DG_y=DG_y_train, batch_size=batch_size,
                                               downsampling_size=hyperparams['downsampling_size'],
                                               mode=hyperparams['mode'], output_format='np.ndarray',
                                               expand_dims=hyperparams['expand_dims'], epochs=epochs,
                                               models_outputs=model.outputs)
            # TODO: following works only in < 2.2 TF versions. Make this work for higher versions.
            # class_weights = self._ae_out_dict(None, class_weights, model.outputs, get_labels=False)
            # print('AE class weights:', class_weights)
        else:
            generator = self.batchGenerator([DG_X_train, DG_y_train], batch_size=batch_size, output_format='np.ndarray',
                                            expand_dims=hyperparams['expand_dims'], epochs=epochs)
        if isinstance(DG_X_train, DataGenerator):
            tot_train = len(DG_X_train)
        else:
            tot_train = len(list(DG_X_train.values())[0])

        # VALIDATION DATA
        # downsample volume label data
        if '_AE' in hyperparams['arch']:
            downsampled_X_val = self._downsample_volume(X_val, hyperparams['mode'], hyperparams['downsampling_size'])
            if '_vanilla_AE' in hyperparams['arch']:
                validation_data = (X_val, downsampled_X_val[hyperparams['mode']])
            else:
                validation_data = (X_val, self._ae_out_dict(downsampled_X_val, y_val, model.outputs))
        else:
            validation_data = (X_val, y_val)

        # FIT
        start_train = time.time()
        history = model.fit(x=generator, epochs=epochs, validation_data=validation_data,
                            steps_per_epoch=tot_train // batch_size, class_weight=class_weights,
                            callbacks=callbacks_list)

        duration_train = int(time.time() - start_train)
        history = history.history

        # EPOCH SELECTION
        # Select model with minimal validation loss after certain number of epochs.
        val_loss_list = copy.deepcopy(history[monitor])
        min_epoch = 40 if hyperparams['mode'] == 'clin' else 60
        split = min(min_epoch, epochs // 2)
        val_loss_list[0:split] = [1e+6] * split
        epoch_with_min_val_loss = np.argmin(val_loss_list) + 1
        print('Epoch with minimal val loss:', epoch_with_min_val_loss)
        selected_model_name = join(temp_models_dir, 'temp_model_' + str(epoch_with_min_val_loss) + '.h5')
        shutil.copyfile(selected_model_name, model_filename)

        # Delete all non-important models using shutil done
        for f in glob.glob(join(temp_models_dir, '*')):
            os.remove(f)
        del val_loss_list

        # SAVE HISTORY
        history['epoch_with_min_val_loss'] = epoch_with_min_val_loss
        history['duration_train'] = duration_train

        # csv
        with open(csv_logfile, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch_with_min_val_loss', epoch_with_min_val_loss])
            writer.writerow(['duration_train', duration_train])

        # SAVE HISTORY PLOT
        model_filename_wo_path = model_filename.split('/')[-1]
        plot_loss_acc_history(history, epochs=epochs, suptitle=model_filename_wo_path, save_name=plot_filename,
                              val=True, save=True, draw_line=epoch_with_min_val_loss)

        # Load the selected copied model.
        model = load_model(model_filename, compile=compile_model,
                           custom_objects={'GroupConv3D': GroupConv3D,
                                           'CoralOrdinal': coral.CoralOrdinal})

        if 'COMB_CROSS' in hyperparams['arch']:
            model.compile(optimizer=opt,
                          loss=hyperparams['loss'](alpha=hyperparams['alpha'],
                                                   dich_class_threshold=hyperparams['dich_class_threshold'],
                                                   num_classes=hyperparams['num_classes']),
                          metrics=hyperparams['metrics'])
        elif '_OR' in hyperparams['arch']:
            model.compile(optimizer=opt,
                          loss=hyperparams['loss'](hyperparams['num_classes'], importance=importance_weights),
                          metrics=[coral.MeanAbsoluteErrorLabels])

        return model

    def validate(self, DG_X_train, DG_y_train, DG_X_val, DG_y_val,
                 logs_path, models_path, hpr, fold_name='fold', arc_name='test', lrate=None, model_name=None,
                 dwi_pretrained_AE_model_name=None, flair_pretrained_AE_model_name=None,
                 DG_y_pretrain_train=None, DG_y_pretrain_val=None, pretrain_mode=None):
        """
        :param DG_X_train: DataGenerator object with training data
        :param DG_y_train: DataGenerator object with training labels
        :param DG_X_val: DataGenerator object with validation data
        :param DG_y_val: DataGenerator object with validation labels
        :param logs_path: Path to folder where the training logs should be saved
        :param models_path: Path to folder where the models should be saved
        :param hpr: dictionary of hyperparameters-ranges to tune (including batch_size and epochs)
        :param fold_name: String: Fold identifier.
        :param arc_name: Architecture name for internal storage of validation loss in the Pipeline object
        :param lrate: LearningRateSchedule object
        :param model_name: String: Stored model_filename to load directly if is not None. No training will occur.
        :param dwi_pretrained_AE_model_name: Model filename of a stored pretrained DWI AE model.
        :param flair_pretrained_AE_model_name: Model filename of a stored pretrained FLAIR AE model.
        :param DG_y_pretrain_train: DataGenerator object with training labels for pretraining DWI CNN
        :param DG_y_pretrain_val: DataGenerator object with validation labels for pretraining DWI CNN
        :param pretrain_mode: mode for pretraining the DWI CNN, options are:
            'reg' for regression
            'class' for binary classification
            None for not doing pretraining

        :return: A tuple with the best_model, best_history, best_params
        """

        # CHECKS
        # check DataGenerator inputs and expand_dims compatibility:
        if DG_y_train is not None:
            self._test_compt([DG_X_train, DG_y_train], self._default_params['expand_dims'])
        if DG_y_val is not None:
            self._test_compt([DG_X_val, DG_y_val], self._default_params['expand_dims'])
        if DG_y_pretrain_train is not None:
            self._test_compt([DG_X_train, DG_y_pretrain_train], self._default_params['expand_dims'])
        if DG_y_pretrain_val is not None:
            self._test_compt([DG_X_val, DG_y_pretrain_val], self._default_params['expand_dims'])

        # make sure that the hyperparameter (keys) correspond with the self._params
        if not isinstance(hpr, dict):
            raise TypeError('hpr must be a dictionary of hyperparameters names / range list')
        for key in hpr.keys():
            if key not in self._default_params:
                raise ValueError('The parameter ', key, ' is not defined for the model. \
                    To add this parameter, you must change the default parameters with one of the following options: \n \
                    1) Update the default hyperparameters using setParams method. \n \
                    2) Change getModel function input in the architecture file and then reload the model with the default \
                    hyperparameters using setArchitecture method. This is not an option for the current patched Pipeline version.')
        if not isinstance(arc_name, str):
            raise TypeError('arc_name must be a string')

        # check pretraining inputs
        if not ((DG_y_pretrain_train is None and DG_y_pretrain_val is None and pretrain_mode is None) or
                (DG_y_pretrain_train is not None and DG_y_pretrain_val is not None and pretrain_mode is not None)):
            raise ValueError('Set pretraining inputs correctly. For run with pretraining, none of DG_y_pretrain_train, '
                             'DG_y_pretrain_val and pretrain_mode must be None. FOr run without pretraining, all '
                             'DG_y_pretrain_train, DG_y_pretrain_val and pretrain_mode must be None.')

        # HELPER FUNCTIONS
        def nameHash(dic, max_len=10):
            """
            Hash a dictionary to a compact pickle filename

            :param dic = dictionary to hash
            :param max_len = integer indicating maximum key/val hashing length
            """

            if not isinstance(dic, dict):
                raise TypeError('dic must be a dictionary')
            if not isinstance(max_len, int):
                raise TypeError('maxlen must be a positive integer')
            if max_len < 2 or max_len > 10:
                raise ValueError('For effective hashing, maxlen must be an integer between 2 and 10')

            renaming_dict = {
                'arch': '',
                'batch_size': 'bs',
                'clinical_depths_of_layers': 'cln_dp',
                'depths_of_classification_layers': 'cls_dp',
                'dropout_rate': 'dr',
                'embedding_img_dim': 'emb_img',
                'embedding_clin_dim': 'emb_cln',
                'epochs': 'ep',
                'h_output': 'h-out',
                'init_filters': 'flt',
                'kernel_regularizer': 'k_reg',
                'kernel_initializer': 'k_init-',
                'decoder_regularizer': 'd_reg',
                'decoder_dropout_rate': 'd_dr',
                'momentum': 'mmnt',
                'optimizer': '',
                'skips': 'skp',
                'stages': 'stg',
                'mode': '',
                'dwi_flair_clin': 'dfc',
                'decoder_stages': 'd_stg',
                'alpha': 'a'
            }

            name = fold_name + '_'
            for key in sorted(dic.keys()):
                if key == 'dwi_freeze':
                    if dic[key] == 1:
                        name = name + key + '_'
                elif key != 'metrics':  # ignore metrics param
                    # Hash key.
                    shortKey = str(key)
                    if shortKey in renaming_dict:
                        shortKey = renaming_dict[shortKey]
                    if len(shortKey) > max_len:
                        shortKey = shortKey[:max_len]

                    # Hash value
                    shortVal = str(dic[key])
                    shortVal = shortVal.replace(' ', '')
                    shortVal = shortVal.replace('\'', '')
                    shortVal = shortVal.replace('[', '')
                    shortVal = shortVal.replace(']', '')
                    shortVal = shortVal.replace('{', '_')
                    shortVal = shortVal.replace('}', '_')
                    if shortVal in renaming_dict:
                        shortVal = renaming_dict[shortVal]
                    if len(shortVal) > max_len and key != 'arch':
                        shortVal = shortVal[-max_len:]
                    name = name + shortKey + shortVal + '_'
            name = name[:-1]
            return name

        def calcGrid(dic):
            """
            Returns a list of dictionaries resulted from all hpr combinations.
            """

            if not isinstance(dic, dict):
                raise TypeError('dic must be a dictionary')

            dicts = [{}]
            for key in dic.keys():
                new_dicts = []
                for val in dic[key]:
                    for subdic in dicts:
                        new_dict = subdic
                        new_dict[key] = val
                        new_dicts.append(copy.deepcopy(new_dict))
                dicts = copy.deepcopy(new_dicts)
            return dicts

        # INITIALIZATION
        params = self._default_params
        grid = calcGrid(hpr)
        exec('self._validation[\'' + arc_name + '\']={}')  # add architecture name to internal storage
        min_loss = None
        best_params = None
        best_model_name = None
        best_tensorboard_name = None
        num_classes_pretrain = None

        # os.chdir(models_path)

        # INPUT SHAPES DEFINITION
        available_modalities = list(DG_X_train.keys())
        single_ID = list(DG_X_train.values())[0].getIDs()[0]
        shapes = {}
        for modality in available_modalities:
            sample = self.getBatch([DG_X_train[modality]], [single_ID], output_format='np.ndarray')
            shapes[modality] = sample.shape[1:]

            if self._default_params['expand_dims'][0][modality] is not None:
                shapes[modality] = shapes[modality] + (1,)  # additional dimension for no. of channels

        print('Input tensor shapes:')
        for k, v in shapes.items():
            print(k, 'input:', v)

        # VALIDATION DATA PREPROCESSING
        if DG_y_val is None:
            X_val = self.getBatch([DG_X_val], list(DG_X_val.values())[0].getIDs(),
                                  expand_dims=self._default_params['expand_dims'])
            y_val = None
        else:
            X_val, y_val = self.getBatch([DG_X_val, DG_y_val], DG_y_val.getIDs(),
                                         expand_dims=self._default_params['expand_dims'])
        X_val = self.preprocess_all_data(X_val, available_modalities)

        # GET PRETRAIN LABELS
        y_pretrain_val = None
        if DG_y_pretrain_val is not None:
            y_pretrain_val = self.getBatch([DG_y_pretrain_val], DG_y_val.getIDs())

        # GRID SEARCH
        print('Grid-search is now starting... ')
        print('Number of hyperparam combinations:', len(grid))
        for i, comb in enumerate(grid):
            tf.keras.backend.clear_session()

            # Update default hyperparams with the current tuned ones from the grid.
            params.update(comb)
            if DG_y_pretrain_train is not None:
                params_pretrain = {k: v for k, v in params.items()}

            # Create file names.
            fname = nameHash(comb, max_len=9)  # name includes tuning hyperparams
            csv_logfile = join(models_path, fname + '_hist.csv')
            model_filename = join(models_path, fname + '_model.h5')
            plot_dir = make_folder(join(models_path, 'plots'))
            plot_filename = join(plot_dir, fname + '.png')
            tensorboard_name = join(logs_path, fname)

            if DG_y_pretrain_train is not None:
                pretrain_comb = copy.deepcopy(comb)
                pretrain_comb['mode'] = 'dwi'
                fname_pretrain = nameHash(pretrain_comb, max_len=9) + '_pretrain'  # name includes tuning hyperparams
                csv_logfile_pretrain = join(models_path, fname_pretrain + '_hist.csv')
                model_filename_pretrain = join(models_path, fname_pretrain + '_model.h5')
                plot_dir_pretrain = make_folder(join(models_path, 'plots_DWI_pretrain'))
                plot_filename_pretrain = join(plot_dir_pretrain, fname_pretrain + '.png')
                tensorboard_name_pretrain = join(logs_path, 'tensorboard', fname_pretrain)

            # Calculate class weights.
            if params['use_class_weights']:
                y_train = self.getBatch([DG_y_train], DG_y_train.getIDs())
                class_weights = self.get_class_weights(y_train)

                if DG_y_pretrain_train:
                    if pretrain_mode == 'reg':
                        print(colored('No class weights for pretraining used.', 'red'))
                        class_weights_pretrain = None
                        params_pretrain['use_class_weights'] = False
                    else:
                        y_pretrain_train = self.getBatch([DG_y_pretrain_train], DG_y_pretrain_train.getIDs())
                        class_weights_pretrain = self.get_class_weights(y_pretrain_train)
            else:
                print(colored('No class weights used.', 'red'))
                class_weights = None
                class_weights_pretrain = None

            # Calculate importance weights for coral loss.
            if params['use_importance_weights']:
                y_train = self.getBatch([DG_y_train], DG_y_train.getIDs())
                importance_weights = self.get_importance_weights(y_train)
            else:
                print(colored('No importance weights used.', 'red'))
                importance_weights = None

            # Load model directly if model_name is not None.
            if model_name is not None and os.path.isfile(model_name):
                model_filename = model_name

            # Check if val loss in internal storage.
            if arc_name in self._validation and fname in self._validation[arc_name]:
                # if val. loss is stored in internal storage
                print('Val loss is already stored in internal storage.')
                temp_loss = self._validation[arc_name][fname]
            else:
                # LOAD OR TRAIN MODEL
                decay_rate = params['lr'] / params['epochs']
                print('')
                print('Optimizer: ', params['optimizer'])
                print('Initial Learning Rate: ', params['lr'])
                print('Momentum: ', params['momentum'])
                print('Decay Rate: ', decay_rate)

                # Set optimizer.
                if params['optimizer'] == 'ADAM':
                    opt = tf.keras.optimizers.Adam(learning_rate=params['lr'], beta_1=params['momentum'],
                                                   decay=decay_rate)
                elif params['optimizer'] == 'RMSPROP':
                    # as originally deviced
                    print('RMSprop not using momentum')
                    opt = tf.keras.optimizers.RMSprop(learning_rate=params['lr'], decay=decay_rate)
                elif params['optimizer'] == 'SGD':
                    opt = tf.keras.optimizers.SGD(learning_rate=params['lr'], momentum=params['momentum'],
                                                  nesterov=params['nesterov'], decay=decay_rate)
                else:
                    raise NotImplementedError('Optimizer <' + params['optimizer'] + '> not implemented')

                compile_model = ('_OR' not in params['arch']) and ('COMB_CROSS' not in params['arch'])

                # Check if model was already trained and saved.
                if os.path.isfile(model_filename):
                    # LOAD MODEL
                    print(colored('Model is already calculated and saved.', 'cyan'))
                    print('The following param. combination is now being loaded:')
                    for item in params.items():
                        print(item)

                    model = load_model(model_filename, compile=compile_model,
                                       custom_objects={'GroupConv3D': GroupConv3D,
                                                       'CoralOrdinal': coral.CoralOrdinal})

                    # Compile model with respective optimizer.
                    if 'COMB_CROSS' in params['arch']:
                        model.compile(optimizer=opt,
                                      loss=params['loss'](alpha=params['alpha'],
                                                          dich_class_threshold=params['dich_class_threshold'],
                                                          num_classes=params['num_classes']),
                                      metrics=params['metrics'])
                    if '_OR' in params['arch']:
                        model.compile(optimizer=opt,
                                      loss=params['loss'](params['num_classes'], importance=importance_weights),
                                      metrics=[coral.MeanAbsoluteErrorLabels])
                else:
                    # BUILD AND TRAIN MODEL
                    # Check whether to do pretraining or not.
                    if DG_y_pretrain_train is not None:
                        # Check if pretrained model was already trained and saved.
                        if os.path.isfile(model_filename_pretrain):
                            # LOAD PRETRAINED MODEL
                            print(colored('Pretrained DWI model is already calculated and saved.', 'cyan'))
                            print('The following param. combination is now being loaded:')
                            for item in params.items():
                                print(item)

                            dwi_model = load_model(model_filename_pretrain, compile=compile_model,
                                                   custom_objects={'GroupConv3D': GroupConv3D,
                                                                   'CoralOrdinal': coral.CoralOrdinal})
                        else:
                            # BUILD AND PRETRAIN DWI CNN
                            print('')
                            print('Pretraining DWI CNN...')
                            params_pretrain['mode'] = 'dwi'
                            if pretrain_mode == 'reg':
                                params_pretrain['loss'] = tf.keras.losses.mean_squared_error
                                params_pretrain['metrics'] = ['mae']
                            elif pretrain_mode == 'class':
                                if num_classes_pretrain > 2:
                                    params_pretrain['loss'] = tf.keras.losses.sparse_categorical_crossentropy
                                    params_pretrain['metrics'] = ['acc']
                                else:
                                    params_pretrain['loss'] = tf.keras.losses.binary_crossentropy
                                    params_pretrain['metrics'] = [tf.keras.metrics.AUC(), 'acc']
                            else:
                                raise ValueError(
                                    'pretrain_mode can be set only to "reg" for regression or "class" for '
                                    'binary classification.')
                            logging.debug('pretraining params: %s', params_pretrain)
                            print('The following pretrain param. combination is now being trained:')
                            for item in params_pretrain.items():
                                print(item)

                            dwi_model = build_dwi_model(dwi_input_shape=shapes['dwi'], hyperparams=params_pretrain,
                                                        num_classes=num_classes_pretrain,
                                                        gridsearch_idx=i, pretrain_mode=pretrain_mode)
                            dwi_model.compile(optimizer=opt, loss=params_pretrain['loss'],
                                              metrics=params_pretrain['metrics'])

                            dwi_model = self.train(DG_X_train, DG_y_pretrain_train, X_val, y_pretrain_val,
                                                   batch_size=params['batch_size'], epochs=params['epochs'],
                                                   model=dwi_model, class_weights=class_weights_pretrain,
                                                   importance_weights=None, csv_logfile=csv_logfile_pretrain,
                                                   tensorboard_name=tensorboard_name_pretrain,
                                                   model_filename=model_filename_pretrain,
                                                   models_path=models_path, hyperparams=params_pretrain, lrate=lrate,
                                                   plot_filename=plot_filename_pretrain, foldname=fold_name, opt=opt,
                                                   compile_model=compile_model)

                        # Build main model and transfer weights from pretrained DWI CNN.
                        print('Training main model with pretrained DWI CNN...')
                        print('The following param. combination is now being trained:')
                        for item in params.items():
                            print(item)
                        model = build_model(input_shapes=shapes, hyperparams=params,
                                            num_classes=params['num_classes'], gridsearch_idx=i,
                                            trained_dwi_model=dwi_model)
                    else:
                        if '_pretrained' in params['arch']:
                            # Build main model and transfer weights from pretrained AE.
                            print('Training main model with pretrained AE...')
                            print('The following param. combination is now being trained:')
                            for item in params.items():
                                print(item)

                            dwi_ae_model = None
                            flair_ae_model = None
                            if 'dwi' in params['mode']:
                                dwi_ae_model = load_model(dwi_pretrained_AE_model_name, compile=compile_model,
                                                          custom_objects={'GroupConv3D': GroupConv3D,
                                                                          'CoralOrdinal': coral.CoralOrdinal})
                            if 'flair' in params['mode']:
                                flair_ae_model = load_model(flair_pretrained_AE_model_name, compile=compile_model,
                                                            custom_objects={'GroupConv3D': GroupConv3D,
                                                                            'CoralOrdinal': coral.CoralOrdinal})

                            model = build_model(input_shapes=shapes, hyperparams=params,
                                                num_classes=params['num_classes'], gridsearch_idx=i,
                                                trained_dwi_ae_model=dwi_ae_model,
                                                trained_flair_ae_model=flair_ae_model,
                                                connect_residuals=params['connect_residuals'])
                        else:
                            # Build main model without transfer learning.
                            print('')
                            print('Training main model without pretraining...')
                            print('The following param. combination is now being trained:')
                            for item in params.items():
                                print(item)
                            if '_vanilla_AE' in params['arch']:
                                model = build_ae_vanilla_model(vol_input_shape=shapes[available_modalities[0]],
                                                               hyperparams=params, gridsearch_idx=i,
                                                               connect_residuals=params['connect_residuals'])
                            else:
                                model = build_model(input_shapes=shapes, hyperparams=params,
                                                    num_classes=params['num_classes'], gridsearch_idx=i,
                                                    connect_residuals=params['connect_residuals'])

                    # Compile main model with respective optimizer and loss.
                    if 'COMB_CROSS' in params['arch']:
                        model.compile(optimizer=opt,
                                      loss=params['loss'](alpha=params['alpha'],
                                                          dich_class_threshold=params['dich_class_threshold'],
                                                          num_classes=params['num_classes']),
                                      metrics=params['metrics'])
                    elif '_OR' in params['arch']:
                        model.compile(optimizer=opt,
                                      loss=params['loss'](params['num_classes'], importance=importance_weights),
                                      metrics=[coral.MeanAbsoluteErrorLabels])
                    else:
                        if '_AE' in params['arch'] and '_vanilla_AE' not in params['arch']:
                            magnitude = 0.001 if 'REG' in params['arch'] else 1
                            losses, loss_weights = ae_loss(alpha=params['alpha'], base_loss=params['loss'],
                                                           reconstruction_loss=params['recon_loss'],
                                                           model_outputs=model.outputs, magnitude=magnitude)
                            model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights,
                                          metrics=params['metrics'])
                        else:
                            model.compile(optimizer=opt, loss=params['loss'], metrics=params['metrics'])

                    # Train main model.
                    model = self.train(DG_X_train, DG_y_train, X_val, y_val, batch_size=params['batch_size'],
                                       epochs=params['epochs'], model=model, class_weights=class_weights,
                                       importance_weights=importance_weights, csv_logfile=csv_logfile,
                                       tensorboard_name=tensorboard_name, model_filename=model_filename,
                                       models_path=models_path, hyperparams=params, lrate=lrate,
                                       plot_filename=plot_filename, foldname=fold_name, opt=opt,
                                       compile_model=compile_model)

                # CHOOSING BEST MODEL IN THE GRID SEARCH
                # Get val_loss on model at checkpoint.
                print('Evaluating model...')
                if '_AE' in params['arch']:
                    downsampled_X_val = self._downsample_volume(X_val, params['mode'], params['downsampling_size'])
                    if '_vanilla_AE' in params['arch']:
                        validation_data = (X_val, downsampled_X_val[params['mode']])
                    else:
                        validation_data = (X_val, self._ae_out_dict(downsampled_X_val, y_val, model.outputs))
                else:
                    validation_data = (X_val, y_val)
                temp_loss_metrics = model.evaluate(*validation_data)
                print("Evaluation loss and metrics:", temp_loss_metrics)
                # for AE architecture select the loss for the classification/regression output only
                if '_AE' in params['arch'] and '_vanilla_AE' not in params['arch']:
                    temp_loss = temp_loss_metrics[1]
                else:
                    temp_loss = temp_loss_metrics[0]
                self._validation[arc_name][fname] = temp_loss

            logging.debug('>>>> MIN LOSS: %s', min_loss)
            logging.debug(best_params)
            logging.debug(best_model_name)

            if not min_loss or temp_loss < min_loss:
                min_loss = temp_loss
                best_params = {k: v for k, v in params.items()}
                best_model_name = model_filename
                best_tensorboard_name = tensorboard_name

            print('MIN LOSS:', min_loss)
            logging.debug(best_params)
            logging.debug(best_model_name)

            print(i + 1, '/', len(grid), 'of grid-search in', fold_name, 'is done.')

            tf.keras.backend.clear_session()

        best_model = load_model(best_model_name, compile=False,
                                custom_objects={'GroupConv3D': GroupConv3D,
                                                'CoralOrdinal': coral.CoralOrdinal})

        print('Validation is done!')

        return best_model, best_params, best_model_name, best_tensorboard_name

    def calcPerformance(self, gt, pred_probs=None, pred_labels=None, metrics=['acc', 'auc', 'roc'],
                        dichotomize_results=False, dicho_class_threshold=0, positive_above_threshold=True,
                        get_calibrated_threshold=False, threshold_calibration_method=None):
        """
        Calculate the performance regarding the given ground_truth and prediction.

        :param gt: Ground truth data, type Tensor or np.array.
        :param pred_probs: Predicted label probabilities, type Tensor or np.array.
        :param pred_labels: Predicted labels.
        :param metrics: List of metrics to compute if use_model_metrics is False, excepts only strings
            (recognition limited to certain measures)
        :param dichotomize_results: Boolean. True for calculate also metrics for dichotomized results.
        :param dicho_class_threshold: Integer in range 0 to 6. Class at which to dichotomize the outcome. The negative
            class includes this class. This is ignored when dichotomize_results is False.
        :param positive_above_threshold: True for the values above the dichotomization threshold represent the positive
        class.
        :param get_calibrated_threshold: True for calibrate sensitivity-specificity-cutoff. False for cutoff set to 0.5.
        :param threshold_calibration_method: Set method when get_calibrated_threshold is True.
            Options: 'youden', {'sensitivity': 0 < x < 1]}, {'specificity': 0 < x < 1}

        :return: Dictionary of computed metrics (dict.keys() same shape as metrics) with float type if return_tensor
            is False, Tensor otherwise
        """

        import tensorflow.keras.metrics as kmetrics
        from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
        from tensorflow.keras.utils import to_categorical

        # Check values.
        if self._default_params['num_classes'] < 2:
            raise ValueError('The number of classes is smaller then 2, but should be >=2.')
        if pred_probs is None and pred_labels is None:
            raise ValueError('Both pred_probs nad pred_labels are empty. Pass one of them to the calsPerformance().')
        if get_calibrated_threshold:
            if threshold_calibration_method != 'youden' and not isinstance(threshold_calibration_method, dict):
                raise ValueError('The value of threshold calibration method argument in calcPerformance function '
                                 'must be either "youden" or a dictionary with the key "sensitivity" or "specificity".')
            if isinstance(threshold_calibration_method, dict):
                if len(threshold_calibration_method.keys()) != 1:
                    raise ValueError('The threshold calibration method argument as dictionary can have just one '
                                     'key-value pair, with key equal to "sensitivity" or "specificity".')
                if list(threshold_calibration_method.keys())[0] != 'sensitivity' and \
                        list(threshold_calibration_method.keys())[0] != 'specificity':
                    raise KeyError('The key in threshold calibration method argument dictionary must be equal to '
                                   '"sensitivity" or "specificity".')
                if list(threshold_calibration_method.values())[0] <= 0 or \
                        list(threshold_calibration_method.values())[0] >= 1:
                    raise ValueError('The value in threshold calibration method argument dictionary must be between '
                                     '0 and 1.')

        # Dimensionality checks
        if pred_probs is not None:
            if self._default_params['num_classes'] > 2:
                gt_onehot = to_categorical(gt)
                if gt_onehot.shape != pred_probs.shape:
                    raise DimensionsMismatch('The dimensions of ground truth data ' + str(gt_onehot.shape) +
                                             ' do not match the dimensions of predictions ' + str(
                        pred_probs.shape) + '.')
            else:
                pred_probs = np.squeeze(pred_probs)
                if gt.shape != pred_probs.shape:
                    raise DimensionsMismatch('The dimensions of ground truth data ' + str(gt.shape) +
                                             ' do not match the dimensions of predictions ' + str(
                        pred_probs.shape) + '.')
        else:
            pred_labels = np.squeeze(pred_labels)
            if gt.shape != pred_labels.shape:
                raise DimensionsMismatch('The dimensions of ground truth data ' + str(gt.shape) +
                                         ' do not match the dimensions of predictions ' + str(
                    pred_labels.shape) + '.')

        def replace_str_to_metric(list_of_metric_names):
            list_of_metrics = []
            new_list_of_metric_names = copy.deepcopy(list_of_metric_names)
            if self._default_params['num_classes'] > 2:
                for metric_name in list_of_metric_names:
                    if metric_name in ['accuracy', 'acc']:
                        list_of_metrics.append(kmetrics.Accuracy())
                    elif metric_name == 'auc':
                        list_of_metrics.append(kmetrics.AUC())
                    elif metric_name in ['sklearn_auc', 'skl_auc']:
                        list_of_metrics.append(roc_auc_score)
                    elif metric_name in ['f1_score', 'f1']:
                        list_of_metrics.append(f1_score)
                    elif metric_name in ['balanced_accuracy', 'bacc']:
                        list_of_metrics.append(balanced_accuracy_score)
                    elif metric_name in ['mean_absolute_error', 'mae']:
                        list_of_metrics.append(kmetrics.MeanAbsoluteError())
                    elif metric_name in ['root_mean_squared_error', 'rmse']:
                        list_of_metrics.append(kmetrics.RootMeanSquaredError())
                    else:
                        print(colored('Warning: Metric type \'' + metric_name +
                                      '\' not supported (will be excluded from calculation).', 'red'))
                        new_list_of_metric_names.remove(metric_name)
            else:
                for metric_name in list_of_metric_names:
                    if metric_name in ['accuracy', 'acc']:
                        list_of_metrics.append(kmetrics.Accuracy())
                    elif metric_name == 'auc':
                        list_of_metrics.append(kmetrics.AUC())
                    elif metric_name in ['sklearn_auc', 'skl_auc']:
                        list_of_metrics.append(roc_auc_score)
                    elif metric_name == 'roc':
                        list_of_metrics.append(kmetrics.AUC())
                    elif metric_name in ['f1_score', 'f1']:
                        list_of_metrics.append(f1_score)
                    elif metric_name in ['balanced_accuracy', 'bacc']:
                        list_of_metrics.append(balanced_accuracy_score)
                    else:
                        print(colored('Warning: Metric type \'' + metric_name +
                                      '\' not supported (will be excluded from calculation).', 'red'))
                        new_list_of_metric_names.remove(metric_name)

            return list_of_metrics, new_list_of_metric_names

        # Convert metric string to functions.
        metrics, metrics_names = replace_str_to_metric(metrics)

        # Get classes from predicted probabilities.
        threshold = None
        if pred_probs is not None:
            if self._default_params['num_classes'] > 2:
                classes = np.argmax(pred_probs, axis=-1).astype(np.uint8)
            else:
                if get_calibrated_threshold:
                    if threshold_calibration_method == 'youden':
                        threshold = sensivity_specifity_cutoff(gt, pred_probs)
                        print(colored('Calibrated classification threshold using Youden\'s index: ' + str(threshold) +
                                      '.', 'yellow'))
                    elif list(threshold_calibration_method.keys())[0] == 'sensitivity':
                        threshold = sensivity_calibrated_cutoff(gt, pred_probs,
                                                                threshold_calibration_method['sensitivity'])
                        print(colored('Calibrated classification threshold to sensitivity of ' +
                                      str(threshold_calibration_method['sensitivity']) + ': ' + str(threshold) + '.',
                                      'yellow'))
                    elif list(threshold_calibration_method.keys())[0] == 'specificity':
                        threshold = specifity_calibrated_cutoff(gt, pred_probs,
                                                                threshold_calibration_method['specificity'])
                        print(colored('Calibrated classification threshold to specificity of ' +
                                      str(threshold_calibration_method['specificity']) + ': ' + str(threshold) + '.',
                                      'yellow'))
                    else:
                        raise ValueError
                else:
                    threshold = 0.5
                    print(colored('Classification threshold set to 0.5.', 'yellow'))
                classes = (pred_probs >= threshold).astype(np.uint8)
        else:
            classes = pred_labels
            if isinstance(classes, np.integer):
                classes = np.around(classes)

        # Loop through metrics and gather outputs
        results_dict = dict()
        for name, metric in zip(metrics_names, metrics):
            # print(name, metric)
            try:
                if self._default_params['num_classes'] > 2:
                    if name == 'auc':  # both gt and pred in one-hot
                        if pred_probs is not None:
                            results_dict[name] = float(metric(gt_onehot, pred_probs))
                    elif name in ['sklearn_auc', 'skl_auc']:  # both gt and pred in one-hot
                        if pred_probs is not None:
                            results_dict[name] = float(
                                metric(gt_onehot, pred_probs, average='macro', multi_class='ovr'))
                    elif name in ['f1_score', 'f1']:
                        results_dict[name] = float(metric(gt, classes, average='macro'))
                    else:
                        results_dict[name] = float(metric(gt, classes))
                else:
                    if name in ['auc', 'sklearn_auc', 'skl_auc']:
                        if pred_probs is not None:
                            results_dict[name] = float(metric(gt, pred_probs))
                    elif name == 'roc':
                        if pred_probs is not None:
                            metric(gt, pred_probs)
                            n = metric.false_positives.read_value() + metric.true_negatives.read_value()
                            p = metric.true_positives.read_value() + metric.false_negatives.read_value()
                            fpr = metric.false_positives.read_value() / n
                            tpr = metric.true_positives.read_value() / p
                            results_dict[name] = (np.array(fpr, dtype=np.float), np.array(tpr, dtype=np.float))
                    else:
                        results_dict[name] = float(metric(gt, classes))
            except ValueError:
                print(colored('Skipping ' + name + ' metric due error ValueError', 'red'))

        if dichotomize_results:
            if self._default_params['num_classes'] <= 2:
                print(colored('Dichotomization of final prediction skipped because the number of classes '
                              'is <=2.', 'red'))
            else:
                if positive_above_threshold:
                    d_gt_int = (gt > dicho_class_threshold) * 1
                    d_prediction = np.clip(pred_probs[:, dicho_class_threshold + 1:].sum(axis=1), 0,
                                           1) if pred_probs is not None else None
                    d_classes = (classes > dicho_class_threshold) * 1
                else:
                    d_gt_int = (gt < dicho_class_threshold) * 1
                    d_prediction = np.clip(pred_probs[:, dicho_class_threshold + 1:].sum(axis=1), 0,
                                           1) if pred_probs is not None else None
                    d_classes = (classes < dicho_class_threshold) * 1
                logging.debug('dicho gt: %s', d_gt_int)
                logging.debug('dicho prediction: %s', d_prediction)
                logging.debug('dicho classes: %s', d_classes)

                for name, metric in zip(metrics_names, metrics):
                    logging.debug(name + ' %s', metric)
                    d_name = 'd_' + name

                    if isinstance(metric, kmetrics.Metric):
                        metric.reset_states()

                    if name in ['auc', 'sklearn_auc', 'skl_auc']:
                        if pred_probs is not None:
                            results_dict[d_name] = float(metric(d_gt_int, d_prediction))
                    else:
                        results_dict[d_name] = float(metric(d_gt_int, d_classes))

        return results_dict, threshold

    def calcPerformance_volumes(self, gt, preds=None, metrics=['mae']):
        """
        Calculate the performance regarding the given ground_truth and prediction.

        :param gt: Ground truth data, type Tensor or np.array.
        :param preds: Linear predictions, type Tensor or np.array.
        :param metrics: List of metrics to compute if use_model_metrics is False, excepts only strings
            (recognition limited to certain measures)

        :return: Dictionary of computed metrics (dict.keys() same shape as metrics) with float type if return_tensor
            is False, Tensor otherwise
        """

        import tensorflow.keras.metrics as kmetrics

        # Dimensionality checks
        if gt.shape != preds.shape:
            raise DimensionsMismatch('The dimensions of ground truth data ' + str(gt.shape) +
                                     ' do not match the dimensions of predictions ' + str(preds.shape) + '.')

        def replace_str_to_metric(list_of_metric_names):
            list_of_metrics = []
            new_list_of_metric_names = copy.deepcopy(list_of_metric_names)

            for metric_name in list_of_metric_names:
                if metric_name in ['mean_absolute_error', 'mae']:
                    list_of_metrics.append(kmetrics.MeanAbsoluteError())
                elif metric_name in ['root_mean_squared_error', 'rmse']:
                    list_of_metrics.append(kmetrics.RootMeanSquaredError())
                else:
                    print(colored('Warning: Metric type \'' + metric_name +
                                  '\' not supported (will be excluded from calculation).', 'red'))
                    new_list_of_metric_names.remove(metric_name)
            return list_of_metrics, new_list_of_metric_names

        # Convert metric string to functions.
        metrics, metrics_names = replace_str_to_metric(metrics)

        # Loop through metrics and gather outputs
        results_dict = dict()
        for name, metric in zip(metrics_names, metrics):
            # print(name, metric)
            try:
                results_dict[name] = float(metric(gt, preds))
            except ValueError:
                print(colored('Skipping ' + name + ' metric due error ValueError', 'red'))

        return results_dict


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)

    pipe = Pipeline({})

    # %%%     Test _ae_labels_dict method     %%%

    dummy_input = tf.keras.layers.Input(shape=(192, 192, 50, 2), name='embedding')
    dummy_out1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                       name='final_output_sigmoid')(dummy_input)
    dummy_out2 = tf.keras.layers.Dense(2, activation='linear',
                                       name='dwi_decoder_output_linear')(dummy_input)
    dummy_out3 = tf.keras.layers.Dense(2, activation='linear',
                                       name='flair_decoder_output_linear')(dummy_input)
    model = tf.keras.models.Model(dummy_input, [dummy_out1, dummy_out2, dummy_out3], name='dummy_model')
    model.summary(line_length=150)

    x = {'clin': ['clin', 'clin'],
         'dwi': ['dwi', 'dwi'],
         'flair': ['flair', 'flair']}
    y = ['y', 'y']

    label_dict = pipe._ae_out_dict(x=x, y=y, model_outputs=model.outputs)
    print('TEST self._ae_labels_dict:', label_dict)

    # %%%     TEST get_class_weights function      %%%

    y = [0, 0, 0, 1]
    pipe.setParams({'num_classes': 2})
    print(pipe.get_class_weights(y))

    z = [0, 0, 0, 2, 1, 1, 1, 0, 2]
    pipe.setParams({'num_classes': 3})
    print(pipe.get_class_weights(z))

    z = [0, 0, 1, 1, 2, 3, 4, 5, 0, 3]
    pipe.setParams({'num_classes': 6})
    print(pipe.get_class_weights(z))

    # %%%     TEST calcPerformance function      %%%

    # binary
    gt = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    pred = np.array([0, 1, 1, 1, 0.3, 0.6, 0.4, 0.7])
    pipe.setParams({'num_classes': 2})
    results = pipe.calcPerformance(gt, pred_probs=pred, metrics=['roc', 'accuracy', 'auc', 'sklearn_auc',
                                                                 'precision', 'recall', 'f1_score',
                                                                 'balanced_accuracy'])
    print(results)

    # multiclass
    gt = np.array([0, 1, 2, 1, 0, 2, 0, 1])
    pred = np.array([[0.7, 0.3, 0],
                     [0.1, 0.3, 0.6],
                     [0.3, 0.6, 0.1],
                     [0.2, 0.5, 0.3],
                     [0.4, 0.25, 0.35],
                     [0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])
    pipe.setParams({'num_classes': 3})
    results = pipe.calcPerformance(gt, pred_probs=pred, metrics=['roc', 'accuracy', 'auc', 'sklearn_auc',
                                                                 'precision', 'recall', 'f1_score',
                                                                 'balanced_accuracy'],
                                   dichotomize_results=True, dicho_class_threshold=1)
    print(results)

    gt = np.array([0, 1, 2, 1, 0, 2, 0, 1])
    pred = np.array([0, 2, 1, 1, 0, 2, 0, 1])
    pipe.setParams({'num_classes': 3})
    results = pipe.calcPerformance(gt, pred_labels=pred, metrics=['roc', 'accuracy', 'auc', 'sklearn_auc',
                                                                  'precision', 'recall', 'f1_score',
                                                                  'balanced_accuracy'],
                                   dichotomize_results=True, dicho_class_threshold=1)
    print(results)

    gt = np.array([849, 185, 663, 109, 62, 308, 916, 199, 73, 115])
    pred = np.array([694, 205.3157, 393.899, 604.84186, 583.3182, 600.41113, 601.9043, 603.17163, 97.18243, 371.6676])
    pipe.setParams({'num_classes': 1000})
    results = pipe.calcPerformance(gt, pred_labels=pred, metrics=['roc', 'accuracy', 'auc', 'sklearn_auc',
                                                                  'precision', 'recall', 'f1_score',
                                                                  'balanced_accuracy', 'mae', 'rmse'],
                                   dichotomize_results=True, dicho_class_threshold=270,
                                   positive_above_threshold=False)
    print(results)

    # %%%     TEST volume resize function      %%%

    import nibabel as nib

    nii = nib.load(join('data/1000plus_coregistered/dwi+flair', '1kplus0006_dwi.nii.gz'))
    volume = nii.get_fdata()
    print('orig shape:', volume.shape)

    down_size = (96, 96, 48)
    resized = resize(volume, (*down_size, volume.shape[-1]))
    print('resized volume shape:', resized.shape)

    nib.save(nib.Nifti1Image(resized, np.eye(4)), join('data/1000plus_coregistered/dwi+flair',
                                                       '1kplus0006_dwi_resized.nii.gz'))

    cut = volume[:, :, 1:-1, :]
    print('cut shape:', cut.shape)

    # scikit
    import time

    start = time.time()
    scikit_cut_resized = resize(cut, (*down_size, cut.shape[-1]))
    print('scikit took:', time.time() - start)
    print('scikit cut resized volume shape:', scikit_cut_resized.shape)

    nib.save(nib.Nifti1Image(scikit_cut_resized, np.eye(4)), join('data/1000plus_coregistered/dwi+flair',
                                                                  '1kplus0006_dwi_cut_resized_scikit.nii.gz'))

    # slicing
    start = time.time()
    halved_cut_resized = cut[::2, ::2]
    print('slicing took:', time.time() - start)
    print('slicing cut resized volume shape:', halved_cut_resized.shape)

    nib.save(nib.Nifti1Image(halved_cut_resized, np.eye(4)), join('data/1000plus_coregistered/dwi+flair',
                                                                  '1kplus0006_dwi_cut_resized_slicing.nii.gz'))
