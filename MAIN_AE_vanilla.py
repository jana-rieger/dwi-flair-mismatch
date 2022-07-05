"""
Train a vanilla Autoencoder for the 3D medical image reconstruction.
This s implemented either for the DWI or for FLAIR modality.
"""

############################################
# Import relevant modules and classes
############################################
import argparse
import json
import logging
import pickle
from datetime import datetime
from os import chdir
from os.path import join, exists

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from termcolor import colored

from ADT.PiplineADT import Pipeline
from ADT.helpers import make_folder


def main():
    ############################################
    # Initiation configuration
    ############################################

    # devs = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(devs[-1], 'GPU')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # logging.basicConfig(level=logging.DEBUG)

    SAVE_DESCRIPTION = 'test_run_vanilla_AE_pretraining'

    SERVER = 'HPCresearch'  # options: 'DL2', 'HPCresearch', 'local'
    DATABASE = '1000plus'  # options: '1000plus', 'HDB'

    ### PARSE COMMAND LINE ARGUMENTS ###

    # MODE options: 'dwi' OR 'flair'
    # ARC_NAME options: groupnet, convnet with '_vanilla_AE' suffix, e.g convnet_vanilla_AE
    # FOLD NUMBER options: the fold number(s) for the current run

    MODE, ARC_NAME, FOLD_NUMBERS = parse_commandline_args()

    if MODE != 'dwi' and MODE != 'flair':
        raise NameError('Unknown modality: ' + MODE + '.' + ' Options are only: "dwi" or "flair".')

    # Set loss function for autoencoder:
    LOSS = tf.keras.losses.mean_squared_error
    METRICS = ['mae']
    FINAL_METRICS = ['mae', 'rmse']

    # Set save location.
    if SERVER == 'DL2':
        save_path = make_folder(join('xxx'))
    elif SERVER == 'HPCresearch':
        save_path = make_folder(join('/fast', 'users', 'jrieger_c', 'work', 'tti_results', DATABASE,
                                     SAVE_DESCRIPTION, MODE))
    elif SERVER == 'local':
        save_path = make_folder(join(r'C:\Users\jrieger\GITrepos\mmop\results\tti', DATABASE, SAVE_DESCRIPTION, MODE))
    else:
        raise ValueError('Unknown server. Set existing server in MAIN_AE_vanilla.py.')

    PERFORMANCE_CSV_PATH = join(save_path, DATABASE + '_' + MODE + '_' + 'vanilla_AE_pretraining_performance.csv')
    save_path = make_folder(join(save_path, ARC_NAME))
    MODELS_PATH = make_folder(join(save_path, 'models'))
    LOGS_PATH = make_folder(join(save_path, 'logs'))
    RESULTS_PATH = make_folder(join(save_path, 'results'))

    # Set data location.
    if DATABASE == '1000plus':
        if SERVER == 'DL2':
            DUF_img_path = join('xxx')
        elif SERVER == 'HPCresearch':
            DUF_img_path = join('/fast', 'users', 'jrieger_c', 'work', 'mmop_data', '1000plus_coregistered',
                                'dwi+flair_with_unlabeled_192x192x48')
        elif SERVER == 'local':
            DUF_img_path = join('/data/1000plus_coregistered/dwi+flair_with_unlabeled_192x192x48')
        else:
            raise ValueError('Unknown server. Set existing server in MAIN_AE_vanilla.py.')

        if 'adc' == MODE:
            DUF_vol_regex = '1kplus*_adc.nii.gz'
        elif 'dwi' == MODE:
            DUF_vol_regex = '1kplus*_dwi.nii.gz'
        elif 'flair' == MODE:
            DUF_vol_regex = '1kplus*_flair.nii.gz'
        else:
            raise NameError('Unknown modality: ' + MODE)

        ID_cut_vol = [6, 10]

        XVAL_PATH_LABELED = './xval_folds_example.json'
        XVAL_PATH_UNLABELED = './xval_folds_unlabeled_example.json'
    else:
        raise ValueError('Unknown database. Set correct database name in MAIN_AE_vanilla.py, variable "database".')

    FTYPE_IMG = 'nii.gz'

    # Define default hyperparameters and hyperparameters-ranges to tune:
    HYPERPARAMETERS = {'arch': 'groupnet',
                       'mode': 'dwi',
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': LOSS,
                       'metrics': METRICS,  # f1_score,
                       'final_assessment': FINAL_METRICS,
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': False,
                       'use_importance_weights': False,
                       # 'expand_dims': [{'dwi': -1, 'flair': -1}, None],  # for one-channel input
                       'expand_dims': [{'dwi': None, 'flair': None}, None],  # for multi-channel input
                       'stages': [1, 1, 1, 1],
                       'bottleneck': True,
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'init_filters': 8,
                       'dropout_rate': 0.2,
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'embedding_img_dim': 256,
                       'dwi_freeze': False,  # for pretraining of the DWI cnn
                       'decoder_stages': 4,  # for autoencoder
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48),  # down-sampling of the input volumes, x and y halved,
                       # z adapted to be divisible by 2 at least 4 times.
                       'connect_residuals': False
                       }

    notes = ARC_NAME + ' ' + MODE + ' ' + SAVE_DESCRIPTION + ' ' + LOSS.__name__
    if 'adc' in MODE:
        MODE = MODE.replace('adc', 'dwi')
    print('notes:', notes)
    print('mode', MODE)

    # Define hyper params to be fine-tuned
    hpr = {'arch': [ARC_NAME],
           'mode': [MODE],
           'optimizer': ['SGD'],  # SGD, ADAM
           'lr': [0.01],  # 0.001], # 0.0001],  # 1e-3, 1e-2, 1e-4
           'momentum': [0.9],  # 0.9
           'batch_size': [8],
           'epochs': [200],
           'init_filters': [8],
           'dropout_rate': [0],
           'kernel_regularizer': [0, 1e-6],
           'decoder_regularizer': [0],
           'decoder_dropout_rate': [0],
           'decoder_stages': [4]
           }

    print(colored('WARNING: Downsampling hardcoded for input shape (192, 192, 48).', 'red', attrs=['blink']))

    # Initiate a Pipeline object:
    pipe = Pipeline({})

    ############################################
    # Load cross validation splits
    ############################################

    print('Loading fold lists from: ', XVAL_PATH_LABELED, XVAL_PATH_UNLABELED)
    with open(XVAL_PATH_LABELED, 'rb') as file:
        folds_list_labeled = json.load(file)
    with open(XVAL_PATH_UNLABELED, 'rb') as file:
        folds_list_unlabeled = json.load(file)

    # Merge the labeled and unlabeled data.
    folds_list = []
    for i in range(len(folds_list_unlabeled)):
        fold_dict = {}
        for k in folds_list_unlabeled[i].keys():
            fold_dict[k] = list(folds_list_unlabeled[i][k]) + list(folds_list_labeled[i][k])
        folds_list.append(fold_dict)

    ############################################
    # Load DUFs to a DataGenerator objects
    ############################################

    # Create a DataGenerator object storing all of the DUF files
    DG_X_vol = pipe.createDG(DUF_vol_regex, DUF_img_path, ID_cut_vol, FTYPE_IMG)
    print('loaded volumes:', len(DG_X_vol))

    ############################################
    # Prepare dict for saving results.
    ############################################

    # Dicts with rows to be appended to dataframe. First save all tuning params.
    result_table = {}
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    result_table['tuning_params'] = {'database': DATABASE, 'date': dt_string, 'notes': ''}
    # Convert values of tuning hpr to strings.
    result_table['tuning_params'].update({k: str(v) for k, v in hpr.items()})
    logging.debug('Row tuning params: %s', result_table['tuning_params'])

    ############################################
    # Run cross validation
    ############################################

    if FOLD_NUMBERS is not None:
        folds_list = [folds_list[fold_number] for fold_number in FOLD_NUMBERS]
    for fold_i, fold in enumerate(folds_list):
        if FOLD_NUMBERS is not None:
            fold_i = FOLD_NUMBERS[fold_i]
        print('')
        print('Running fold %d started...' % fold_i)

        ############################################
        # Get fold splits
        ############################################

        input_dict = prepare_sets(fold, fold_i, DG_X_vol, MODELS_PATH, MODE)

        ############################################
        # Preprocess
        ############################################

        print("Setting preprocessing statistics...")
        if MODE == 'dwi':
            pipe.setStandartization(DG_X_dwi_train=input_dict['train']['dwi'],
                                    DG_X_flair_train=None,
                                    DG_X_clin_train=None,
                                    numerical_data=[], numerical_data_indices=[])
        elif MODE == 'flair':
            pipe.setStandartization(DG_X_dwi_train=None,
                                    DG_X_flair_train=input_dict['train']['flair'],
                                    DG_X_clin_train=None,
                                    numerical_data=[], numerical_data_indices=[])
        else:
            raise NameError('Unknown modality: ' + MODE)
        print("Setting preprocessing statistics done.")

        ############################################
        # Set default hyperparameters
        ############################################

        pipe.setParams(HYPERPARAMETERS)

        ############################################
        # Validate the model using grid-search
        ############################################

        # Make sure we're in the right folder:
        chdir(RESULTS_PATH)

        best_model, best_params, best_model_name, best_tensorboard_name = \
            pipe.validate(
                DG_X_train=input_dict['train'], DG_y_train=None,
                DG_X_val=input_dict['val'], DG_y_val=None,
                logs_path=LOGS_PATH, models_path=MODELS_PATH, hpr=hpr, arc_name=ARC_NAME,
                fold_name='fold' + str(fold_i), lrate=None, model_name=None
            )

        print('Dataset: ', DATABASE, '- Fold:', fold_i)
        print('The models hyperparameters were tuned to:')
        for item in best_params.items():
            print(item)

        ############################################
        # Assess model performance
        ############################################
        print('Predicting for final assessment...')
        # Dictionaries for predicted probabilities and labels for train, val and test set.
        datasets = ['train', 'val', 'test']
        preds, labels, features = predict_sets(set_list=datasets, pipe=pipe, input_dict=input_dict,
                                               model=best_model, expand_dims=HYPERPARAMETERS['expand_dims'],
                                               mode=HYPERPARAMETERS['mode'],
                                               downsampling_size=HYPERPARAMETERS['downsampling_size'],
                                               ae='_AE' in ARC_NAME and '_vanilla_AE' not in ARC_NAME)

        # Save predictions.
        results_name = best_model_name.replace('_model.h5', '').split('/')[-1]
        for dset in datasets:
            ids = input_dict[dset][HYPERPARAMETERS['mode']].getIDs()
            for i in range(2):
                nifti_image = nib.Nifti1Image(preds[dset][i], affine=np.eye(4))
                nib.save(nifti_image, join(RESULTS_PATH, dset + "_" + ids[i] + "_" + results_name + ".nii.gz"))
                nifti_image = nib.Nifti1Image(labels[dset][i], affine=np.eye(4))
                nib.save(nifti_image, join(RESULTS_PATH, dset + "_" + ids[i] + "_" + "target" + ".nii.gz"))
                nifti_image = nib.Nifti1Image(features[dset][MODE][i], affine=np.eye(4))
                nib.save(nifti_image, join(RESULTS_PATH, dset + "_" + ids[i] + "_" + "features" + ".nii.gz"))

        # Save some notes.
        result_table['fold_' + str(fold_i)] = {'database': DATABASE, 'date': dt_string, 'notes': notes}
        result_table['fold_' + str(fold_i)].update({k: str(best_params[k]) for k, v in hpr.items()})
        results = {}

        # Calculate performance for current fold and save to result_table dict.
        for dset in datasets:
            results[dset] = pipe.calcPerformance_volumes(gt=labels[dset], preds=preds[dset],
                                                         metrics=HYPERPARAMETERS['final_assessment'])

        print('Performance on the', datasets, 'set:')
        for metric_name in results['train']:
            for dset in datasets:
                print(metric_name, ':', results[dset][metric_name])
                result_table['fold_' + str(fold_i)].update({metric_name + '_' + dset: results[dset][metric_name]})

        result_table['fold_' + str(fold_i)].update({'tensorboard_name': best_tensorboard_name.split('/')[-1],
                                                    'model_name': best_model_name.split('/')[-1]})

        logging.debug('Row fold_' + str(fold_i) + ': %s', result_table['fold_' + str(fold_i)])

        tf.keras.backend.clear_session()

    ############################################
    # Get mean and std for metrics
    ############################################

    # Prepare rows with mean and std values.
    result_table['mean'] = {'database': DATABASE, 'date': dt_string}
    result_table['std'] = {'database': DATABASE, 'date': dt_string}

    # Calculate metrics mean and std from the folds.
    metrics_gathered = {'train': {}, 'val': {}, 'test': {}}
    for dset in datasets:
        for metric_name in results['train'].keys():
            metrics_gathered[dset][metric_name] = [result_table['fold_' + str(f_idx)][metric_name + '_' + dset]
                                                   for f_idx in range(len(folds_list))]
            result_table['mean'].update(
                {metric_name + '_' + dset: np.array(metrics_gathered[dset][metric_name]).mean()})
            result_table['std'].update({metric_name + '_' + dset: np.array(metrics_gathered[dset][metric_name]).std()})

        print('')
        try:
            print(dset + ' bACC:', metrics_gathered[dset]['bacc'])
            print('Mean ' + dset + ' bACC: ', result_table['mean']['bacc_' + dset])
            print('STD ' + dset + ' bACC: ', result_table['std']['bacc_' + dset])
        except KeyError:
            pass

    logging.debug('Row mean: %s', result_table['mean'])
    logging.debug('Row std: %s', result_table['std'])

    ############################################
    # Save to dataframe
    ############################################

    if exists(PERFORMANCE_CSV_PATH):
        df1 = pd.read_csv(PERFORMANCE_CSV_PATH, index_col=0)
        df2 = pd.DataFrame.from_dict(result_table, orient='index')
        df = pd.concat([df1, df2])
    else:
        df = pd.DataFrame.from_dict(result_table, orient='index')

    df.to_csv(PERFORMANCE_CSV_PATH)
    print('Performance saved to:', PERFORMANCE_CSV_PATH)


# noinspection PyShadowingNames
def prepare_sets(id_lists, fold_i, DG_X_vol, models_path, mod):
    ############################################
    # Split into training/ validation / test sets:
    ############################################

    DG_X_vol_train = DG_X_vol.ID_split(id_lists['train'])
    DG_X_vol_val = DG_X_vol.ID_split(id_lists['val'])
    DG_X_vol_test = DG_X_vol.ID_split(id_lists['test'])

    print('No. of patients in training set:', len(DG_X_vol_train))
    print('No. of patients in validation set:', len(DG_X_vol_val))
    print('No. of patients in test set:', len(DG_X_vol_test))

    # Save patients in splits
    patients = {'train': list(DG_X_vol_train.getIDs()),
                'val': list(DG_X_vol_val.getIDs()),
                'test': list(DG_X_vol_test.getIDs())}
    with open(join(models_path, 'fold_' + str(fold_i) + 'patient_splits.p'), 'wb') as handle:
        pickle.dump(patients, handle)

    x_dict = {'train':
                  {mod: DG_X_vol_train},
              'val':
                  {mod: DG_X_vol_val},
              'test':
                  {mod: DG_X_vol_test}}

    return x_dict


def predict_sets(set_list, pipe, input_dict, model, expand_dims, mode, downsampling_size, ae=False):
    pred_probs = {}
    labels = {}
    features = {}

    for dset in set_list:
        available_modalities = list(input_dict[dset].keys())
        features[dset] = pipe.getBatch([input_dict[dset]], input_dict[dset][available_modalities[0]].getIDs(),
                                       output_format='np.ndarray', expand_dims=expand_dims)
        features[dset] = pipe.preprocess_all_data(features[dset], available_modalities)
        labels[dset] = pipe._downsample_volume(features[dset], mode, downsampling_size)[mode]

        if ae:
            pred_probs[dset] = model.predict(features[dset])[0]
        else:
            pred_probs[dset] = model.predict(features[dset])

        logging.debug(dset + ' PREDICTION SHAPE: %s', pred_probs[dset].shape)
        logging.debug(dset + ' PREDICTION: %s', pred_probs[dset][:10])

    return pred_probs, labels, features


def parse_commandline_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", help="MODE options: 'dwi' OR 'flair'")
    parser.add_argument("-a", "--arc_name", help="groupnet, convnet with '_vanilla_AE' suffix, e.g convnet_vanilla_AE")
    parser.add_argument("-f", "--fold", help="crossvalidation fold number(s)")

    args = parser.parse_args()

    mode = args.mode
    arc_name = args.arc_name
    if args.fold is not None:
        fold_list = args.fold.split(',')
        fold_numbers = [int(n.strip()) for n in fold_list]
    else:
        fold_numbers = None

    return mode, arc_name, fold_numbers


if __name__ == '__main__':
    main()
