"""
Train a main multi-modal DL model for a classification or regression task.
Excepts modalities: 'clin', 'dwi', 'flair'.
"""

############################################
# Import relevant modules and classes
############################################
import argparse
import copy
import json
import logging
import pickle
from datetime import datetime
from os.path import join, exists

import coral_ordinal as coral  # pip install coral-ordinal
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import special
from termcolor import colored

from ADT.PiplineADT import Pipeline
from ADT.helpers import make_folder, EndswithDict, StartswithDict
from ADT.losses import CombinedCrossentropy


def main():
    ############################################
    # Initiation configuration
    ############################################

    # devs = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(devs[-1], 'GPU')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # logging.basicConfig(level=logging.DEBUG)

    DICHO_THRESHOLD = 270  # 2 mRS category; 270 minutes time to imaging; 4.5 hours time to imaging
    POSITIVE_ABOVE_THRESHOLD = False  # False for regression experiments with TTI, otherwise True
    SAVE_DESCRIPTION = 'test_run'
    GET_CALIBRATED_CLASSIFICATION_THRESHOLD = True
    # Options for calibration method: 'youden'; {'sensitivity': 0 < x < 1}; {'specificity': 0 < x < 1}
    THRESHOLD_CALIBRATION_METHOD = 'youden'

    SERVER = 'HPCresearch'  # options: 'DL2', 'HPCresearch', 'local'
    DATABASE = '1000plus'  # options: '1000plus', 'HDB'

    PRETRAINED_AE_MODELS = {
        'dwi': {
            'fold0':
                '/fast/work/users/jrieger_c/tti_results/1000plus/test_run_vanilla_AE_pretraining/dwi/groupnet_vanilla_AE/models/fold0_groupnet_vanilla_AE_bs8_d_dr0_d_reg0_d_stg4_dr0_ep3_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_model.h5',
            'fold1':
                '/fast/work/users/jrieger_c/tti_results/1000plus/test_run_vanilla_AE_pretraining/dwi/groupnet_vanilla_AE/models/fold1_groupnet_vanilla_AE_bs8_d_dr0_d_reg0_d_stg4_dr0_ep3_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_model.h5',
            'fold2':
                '/fast/work/users/jrieger_c/tti_results/1000plus/test_run_vanilla_AE_pretraining/dwi/groupnet_vanilla_AE/models/fold2_groupnet_vanilla_AE_bs8_d_dr0_d_reg0_d_stg4_dr0_ep3_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_model.h5',
            'fold3':
                '/fast/work/users/jrieger_c/tti_results/1000plus/test_run_vanilla_AE_pretraining/dwi/groupnet_vanilla_AE/models/fold3_groupnet_vanilla_AE_bs8_d_dr0_d_reg0_d_stg4_dr0_ep3_flt8_k_reg1e-06_lr0.01_dwi_mmnt0.9_SGD_model.h5'
        },
        'flair': {
            'fold0':
                'path to file',
            'fold1':
                'path to file',
            'fold2':
                'path to file',
            'fold3':
                'path to file'
        }}

    ### PARSE COMMAND LINE ARGUMENTS ###

    # MODE options: 'dwi', 'flair', 'clin', or combinations e.g. 'dwi_flair_clin', 'adc_flair_clin'
    # ARC_NAME options: groupnet, clinicalNN, convnet
    # add 'OR' to the arch name to indicate the ordinal classification with CORAL, e.g. convnet_OR
    # add 'REG' to the arch name to indicate the standard regression, e.g. convnet_REG
    # add 'COMB_CROSS' to the arch name to indication combined loss of categorical and binary crossentropy,
    # e.g. convnet_COMB_CROSS
    # add 'AE' to the arch name indicate the adding of the decoder path to the given architecture as in autoencoders,
    # e.g convnet_AE or convnet_REG_AE
    # CLIN_FEATURES_SET options: 'dicho_tti'
    # FOLD NUMBER options: the fold number(s) for the current run
    # The following are not relevant for the time-to-MRI study:
    # PRETRAIN_LABEL options: None
    # PRETRAIN_MODE options: None for no pretraining, 'reg' for regression, 'class' for binary classification
    # PRETRAIN_FREEZING options: None for no pretraining, '0' for tuning dwi model, '1' for freezing pretrained dwi
    # model

    MODE, ARC_NAME, CLIN_FEATURES_SET, PRETRAIN_LABEL, PRETRAIN_MODE, PRETRAIN_FREEZING, FOLD_NUMBERS = parse_commandline_args()

    # Selects the number of clinical features for all the feature sets that end with the dict keys.
    num_features = EndswithDict({'tti': 0})

    # !!! The ordering of the keys is important, because the endswith builtin function returns the first match.
    # When keys in conflict, e.g 'dicho_shift_release' and 'release', longer keys must be listed first in the dict.
    numerical_data = EndswithDict({'tti': []})
    numerical_data_indices = EndswithDict({'tti': []})
    # Selects the number of classes for all the clinical feature sets that start with the dict keys.
    num_classes = StartswithDict({'dicho': 2,
                                  'continuous': 1000
                                  })

    NUM_CLASSES = num_classes[CLIN_FEATURES_SET]

    # Select base loss function and metrics.
    if 'COMB_CROSS' in ARC_NAME:
        LOSS = CombinedCrossentropy
        METRICS = ['acc']
        FINAL_METRICS = ['auc', 'skl_auc', 'bacc', 'acc']
    elif 'REG' in ARC_NAME:
        LOSS = tf.keras.losses.mean_absolute_error
        METRICS = ['mae']
        FINAL_METRICS = ['bacc', 'acc', 'mae', 'rmse']
    else:
        if NUM_CLASSES == 2:
            LOSS = tf.keras.losses.binary_crossentropy
            METRICS = ['acc']  # tf.keras.metrics.AUC(),
            FINAL_METRICS = ['auc', 'bacc', 'acc']
        elif NUM_CLASSES > 2:
            if 'OR' in ARC_NAME:
                LOSS = coral.OrdinalCrossEntropy
                METRICS = ['acc']
                FINAL_METRICS = ['auc', 'skl_auc', 'bacc', 'acc']
            else:
                LOSS = tf.keras.losses.sparse_categorical_crossentropy
                METRICS = ['acc']
                FINAL_METRICS = ['auc', 'skl_auc', 'bacc', 'acc']
        else:
            raise ValueError('Cannot define loss. Set correct number of classes in MAIN.py script.')

    # Set loss function for autoencoder:
    RECONSTRUCT_LOSS = tf.keras.losses.mean_squared_error

    # Set save location.
    if SERVER == 'DL2':
        save_path = make_folder(join('xxx'))
    elif SERVER == 'HPCresearch':
        save_path = make_folder(join('/fast', 'users', 'jrieger_c', 'work', 'tti_results', DATABASE,
                                     SAVE_DESCRIPTION, MODE, CLIN_FEATURES_SET))
    elif SERVER == 'local':
        save_path = make_folder(join(r'C:\Users\jrieger\GITrepos\mmop\results\tti', DATABASE, SAVE_DESCRIPTION,
                                     MODE, CLIN_FEATURES_SET))
    else:
        raise ValueError('Unknown server. Set existing server in MAIN.py.')

    PERFORMANCE_CSV_PATH = join(save_path, DATABASE + '_' + MODE + '_' + CLIN_FEATURES_SET + '_performance.csv')
    save_path = make_folder(join(save_path, ARC_NAME))
    MODELS_PATH = make_folder(join(save_path, 'models'))
    LOGS_PATH = make_folder(join(save_path, 'logs'))
    RESULTS_PATH = make_folder(join(save_path, 'results'))

    # Set data location.
    if DATABASE == '1000plus':
        if SERVER == 'DL2':
            DUF_clin_path = join('xxx')
            DUF_img_path = join('xxx')
        elif SERVER == 'HPCresearch':
            DUF_clin_path = join('/fast', 'users', 'jrieger_c', 'work', 'mmop_data', '1000plus_coregistered',
                                 'clinical', CLIN_FEATURES_SET)
            DUF_img_path = join('/fast', 'users', 'jrieger_c', 'work', 'mmop_data', '1000plus_coregistered',
                                'dwi+flair_192x192x48')
        elif SERVER == 'local':
            DUF_clin_path = join('./data/1000plus_coregistered/clinical', CLIN_FEATURES_SET)
            DUF_img_path = join('./data/1000plus_coregistered/dwi+flair_192x192x48')
        else:
            raise ValueError('Unknown server. Set existing server in MAIN.py.')

        DUF_clin_regex = '1000plus_clinData_' + str(num_features[CLIN_FEATURES_SET]) + 'cov_w_header_X*'
        DUF_y_regex = '1000plus_clinData_' + str(num_features[CLIN_FEATURES_SET]) + 'cov_w_header_y*'
        if 'adc' in MODE:
            DUF_dwi_regex = '1kplus*_adc.nii.gz'
        else:
            DUF_dwi_regex = '1kplus*_dwi.nii.gz'
        DUF_flair_regex = '1kplus*_flair.nii.gz'

        ID_cut_clin = [-8, -4]
        ID_cut_dwi = [6, 10]
        ID_cut_flair = [6, 10]

        NUMERICAL_DATA = numerical_data[CLIN_FEATURES_SET]
        NUMERICAL_DATA_INDICES = numerical_data_indices[CLIN_FEATURES_SET]

        XVAL_PATH = './xval_folds_example.json'
    else:
        raise ValueError('Unknown database. Set correct database name in MAIN.py, variable "database".')

    FTYPE_CLIN = 'csv_w_header'
    FTYPE_IMG = 'nii.gz'

    # Define default hyperparameters and hyperparameters-ranges to tune:
    HYPERPARAMETERS = {'arch': 'groupnet',
                       'mode': 'dwi_flair_clin',
                       'activation': 'relu',
                       'kernel_initializer': 'he_normal',
                       'optimizer': 'SGD',
                       'momentum': 0.9,  # only relevant for SGD optimizer
                       'nesterov': True,  # only relevant for SGD optimizer
                       'lr': 0.001,
                       'loss': LOSS,
                       'recon_loss': RECONSTRUCT_LOSS,
                       'metrics': METRICS,  # f1_score,
                       'final_assessment': FINAL_METRICS,
                       'batch_size': 8,
                       'epochs': 150,
                       'use_class_weights': False,
                       'use_importance_weights': False,
                       # 'expand_dims': [{'clin': None, 'dwi': -1, 'flair': -1}, None],  # for one-channel input
                       'expand_dims': [{'clin': None, 'dwi': None, 'flair': None}, None],  # for multi-channel input
                       'num_classes': NUM_CLASSES,
                       'stages': [1, 1, 1, 1],
                       'bottleneck': True,
                       'h_output': 'D4H',  # for GroupNet
                       'kernel_size': 3,
                       'init_filters': 8,
                       'dropout_rate': 0.2,
                       'kernel_regularizer': 0,
                       'skips': False,
                       'use_se': False,
                       'clinical_depths_of_layers': [128, 256, 128],
                       'embedding_img_dim': 256,
                       'embedding_clin_dim': 256,
                       'depths_of_classification_layers': [256],
                       'dich_class_threshold': DICHO_THRESHOLD,
                       'alpha': 0,  # for combined loss
                       'dwi_freeze': False,  # for pretraining
                       'decoder_stages': 4,  # for autoencoder
                       'decoder_regularizer': 0,
                       'decoder_dropout_rate': 0,
                       'tensorboard': True,
                       'downsampling_size': (96, 96, 48),  # down-sampling of the input volumes, x and y halved,
                       # z adapted to be divisible by 2 at least 4 times.
                       'connect_residuals': False
                       }

    notes = ARC_NAME + ' ' + MODE + ' ' + CLIN_FEATURES_SET + ' ' + SAVE_DESCRIPTION + ' ' + LOSS.__name__
    if GET_CALIBRATED_CLASSIFICATION_THRESHOLD:
        notes += ' calibrated class threshold, ' + str(THRESHOLD_CALIBRATION_METHOD)
    if 'adc' in MODE:
        MODE = MODE.replace('adc', 'dwi')
    print('notes:', notes)
    print('mode', MODE)

    # Define hyper params to be fine-tuned
    hpr = {'arch': [ARC_NAME],
           'mode': [MODE],
           'optimizer': ['SGD'],  # SGD, ADAM
           'lr': [0.01, 0.001, 0.0001],  # 1e-3, 1e-2, 1e-4
           'momentum': [0.9],
           'batch_size': [8],
           'epochs': [200],
           'init_filters': [8],
           'dropout_rate': [0.2, 0.3, 0.4],  # the same dropout used in every network (dwi, flair, clin),  0.2  - 0.4
           'kernel_regularizer': [1e-6],
           # 'decoder_regularizer': [0],
           # 'decoder_dropout_rate': [0, 0.2, 0.4],
           # 'alpha': [0.25, 0.5, 0.75],  # tune on selected number of decoder stages
           # 'decoder_stages': [4]
           }

    if PRETRAIN_FREEZING is not None:
        hpr['dwi_freeze'] = [PRETRAIN_FREEZING]

    clin_hpr = {'arch': [ARC_NAME],
                'mode': [MODE],
                'optimizer': ['SGD'],  # try also Adam
                'lr': [0.01, 0.005, 0.001],  # [0.001, 0.0005],  # 1e-3, 1e-2, 1e-4
                'momentum': [0.9],  # 0.9
                'batch_size': [64],
                'epochs': [150],
                'dropout_rate': [0.2, 0.3, 0.4],
                # the same dropout used in every network (dwi, flair, clin),  0.2  - 0.4
                'kernel_regularizer': [1e-6],
                # is not efficient in the imaging part, but it can be efficient in the classification part
                # 'clinical_depths_of_layers': [[128, 256, 128]],
                # 'alpha': [0.25, 0.5, 0.75]
                }

    if 'clin' in ARC_NAME:
        hpr = clin_hpr

    if '_AE' in ARC_NAME:
        print(colored('WARNING: The autoencoder is implemented just for DWI mode.', 'red'))
        print(colored('WARNING: Downsampling hardcoded for input shape (192, 192, 48).', 'red', attrs=['blink']))

    # set here this line to use lrate decay
    lrate = None

    # Initiate a Pipeline object:
    pipe = Pipeline({})

    ############################################
    # Load cross validation splits
    ############################################

    print('Loading from directory: ', XVAL_PATH)
    with open(XVAL_PATH, 'rb') as file:
        folds_list = json.load(file)

    # just for quick debugging
    # with open(XVAL_PATH, 'r') as file:
    #     folds_list = json.load(file)

    ############################################
    # Load DUFs to a DataGenerator objects
    ############################################

    # Create a DataGenerator object storing all of the DUF files
    DG_X_clin = pipe.createDG(DUF_clin_regex, DUF_clin_path, ID_cut_clin, FTYPE_CLIN)
    DG_X_dwi = pipe.createDG(DUF_dwi_regex, DUF_img_path, ID_cut_dwi, FTYPE_IMG)
    DG_X_flair = pipe.createDG(DUF_flair_regex, DUF_img_path, ID_cut_flair, FTYPE_IMG)
    DG_y = pipe.createDG(DUF_y_regex, DUF_clin_path, ID_cut_clin, FTYPE_CLIN)

    DG_y_pretrain = None
    if PRETRAIN_LABEL is not None:
        DG_y_pretrain = pipe.createDG(DUF_y_regex, join(DUF_clin_path, PRETRAIN_LABEL), ID_cut_clin, FTYPE_CLIN)

    print('loaded clinical DUFs:', len(DG_X_clin))
    print('loaded DWIs:', len(DG_X_dwi))
    print('loaded FLAIRs:', len(DG_X_flair))

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

    if 'OR' in ARC_NAME:
        result_table2 = {}
        result_table2['tuning_params'] = {'database': DATABASE, 'date': dt_string, 'notes': 'CORAL labels method 2'}
        # Convert values of tuning hpr to strings.
        result_table2['tuning_params'].update({k: str(v) for k, v in hpr.items()})

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

        input_dict, label_dict, label_pretrain_dict = prepare_sets(fold, fold_i, DG_X_clin, DG_X_dwi, DG_X_flair, DG_y,
                                                                   DG_y_pretrain, MODELS_PATH)

        ############################################
        # Preprocess
        ############################################

        print("Setting preprocessing statistics...")
        pipe.setStandartization(DG_X_dwi_train=input_dict['train']['dwi'],
                                DG_X_flair_train=input_dict['train']['flair'],
                                DG_X_clin_train=input_dict['train']['clin'],
                                numerical_data=NUMERICAL_DATA, numerical_data_indices=NUMERICAL_DATA_INDICES)
        print("Setting preprocessing statistics done.")

        ############################################
        # Set default hyperparameters
        ############################################

        pipe.setParams(HYPERPARAMETERS)

        ############################################
        # Validate the model using grid-search
        ############################################

        best_model, best_params, best_model_name, best_tensorboard_name = \
            pipe.validate(
                DG_X_train=input_dict['train'], DG_y_train=label_dict['train'],
                DG_X_val=input_dict['val'], DG_y_val=label_dict['val'],
                logs_path=LOGS_PATH, models_path=MODELS_PATH, hpr=hpr, arc_name=ARC_NAME,
                fold_name='fold' + str(fold_i), lrate=lrate, model_name=None,
                dwi_pretrained_AE_model_name=PRETRAINED_AE_MODELS['dwi']['fold' + str(fold_i)],
                flair_pretrained_AE_model_name=PRETRAINED_AE_MODELS['flair']['fold' + str(fold_i)],
                DG_y_pretrain_train=label_pretrain_dict['train'], DG_y_pretrain_val=label_pretrain_dict['val'],
                pretrain_mode=PRETRAIN_MODE
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
        preds, labels = predict_sets(set_list=datasets, pipe=pipe, input_dict=input_dict, label_dict=label_dict,
                                     model=best_model, expand_dims=HYPERPARAMETERS['expand_dims'],
                                     ae='_AE' in ARC_NAME)
        preds_labels_method2 = None

        # Convert ordinal logits to probabilities and labels.
        # Note that the predictions by CORAL are ordinal (cumulative) logits, not probabilities or regular logits.
        if 'OR' in ARC_NAME:
            ordinal_logits = copy.deepcopy(preds)
            cum_probs = {}
            preds_labels_method2 = {}
            for dset in datasets:
                # There are two methods of converting the ordinal logits into labels:
                # 1. method: Convert from ordinal logits to label probabilities and then in PipelineADT
                # calcPerformance() function convert the probabilities to labels. Method proposed in Frank and Hall
                # (2001), 'A Simple Approach to Ordinal Classification'.
                preds[dset] = coral.ordinal_softmax(ordinal_logits[dset]).numpy()
                # 2. method: proposed in Cao et al. (2020), 'Rank consistent ordinal regression for
                # neural networks with application to age estimation'. This method returns directly the labels, but no
                # label probabilities so the ROC AUC cannot be calculated.
                cum_probs[dset] = pd.DataFrame(ordinal_logits[dset]).apply(special.expit)
                preds_labels_method2[dset] = cum_probs[dset].apply(lambda x: x > 0.5).sum(axis=1)

        # Save some notes.
        result_table['fold_' + str(fold_i)] = {'database': DATABASE, 'date': dt_string, 'notes': notes}
        result_table['fold_' + str(fold_i)].update({k: str(best_params[k]) for k, v in hpr.items()})
        results = {}
        if 'OR' in ARC_NAME:
            result_table2['fold_' + str(fold_i)] = {'database': DATABASE, 'date': dt_string, 'notes': notes}
            result_table2['fold_' + str(fold_i)].update({k: str(best_params[k]) for k, v in hpr.items()})
            results_method2 = {}
        else:
            results_method2 = None

        # Calculate performance for current fold and save to result_table dict.
        class_thresholds = {}
        for dset in datasets:
            if 'REG' in ARC_NAME:
                results[dset], threshold = pipe.calcPerformance(gt=labels[dset], pred_labels=preds[dset],
                                                                metrics=HYPERPARAMETERS['final_assessment'],
                                                                dichotomize_results=True, dicho_class_threshold=
                                                                HYPERPARAMETERS['dich_class_threshold'],
                                                                positive_above_threshold=POSITIVE_ABOVE_THRESHOLD)
            else:
                results[dset], threshold = pipe.calcPerformance(gt=labels[dset], pred_probs=preds[dset],
                                                                metrics=HYPERPARAMETERS['final_assessment'],
                                                                dichotomize_results=True, dicho_class_threshold=
                                                                HYPERPARAMETERS['dich_class_threshold'],
                                                                get_calibrated_threshold=
                                                                GET_CALIBRATED_CLASSIFICATION_THRESHOLD,
                                                                threshold_calibration_method=
                                                                THRESHOLD_CALIBRATION_METHOD)

            if 'OR' in ARC_NAME:
                # Estimate performance also for the second method of converting the ordinal logits into labels.
                results_method2[dset], threshold = pipe.calcPerformance(gt=labels[dset],
                                                                        pred_labels=preds_labels_method2[dset],
                                                                        metrics=HYPERPARAMETERS['final_assessment'],
                                                                        dichotomize_results=True, dicho_class_threshold=
                                                                        HYPERPARAMETERS['dich_class_threshold'])
            class_thresholds[dset] = threshold
        for dset in datasets:
            result_table['fold_' + str(fold_i)].update({'class_threshold_' + dset: class_thresholds[dset]})

        print('Performance on the', datasets, 'set:')
        for metric_name in results['train']:
            for dset in datasets:
                print(metric_name, ':', results[dset][metric_name])
                result_table['fold_' + str(fold_i)].update({metric_name + '_' + dset: results[dset][metric_name]})

        if 'OR' in ARC_NAME:
            print('Performance OR method 2 on the', datasets, 'set:')
            for metric_name in results_method2['train']:
                for dset in datasets:
                    print(metric_name, ':', results_method2[dset][metric_name])
                    result_table2['fold_' + str(fold_i)].update(
                        {metric_name + '_' + dset: results_method2[dset][metric_name]})

        result_table['fold_' + str(fold_i)].update({'tensorboard_name': best_tensorboard_name.split('/')[-1],
                                                    'model_name': best_model_name.split('/')[-1]})
        if 'OR' in ARC_NAME:
            result_table2['fold_' + str(fold_i)].update({'tensorboard_name': best_tensorboard_name.split('/')[-1],
                                                         'model_name': best_model_name.split('/')[-1]})

        logging.debug('Row fold_' + str(fold_i) + ': %s', result_table['fold_' + str(fold_i)])

        # Save predictions.
        results_name = best_model_name.replace('_model.h5', '').split('/')[-1]
        if 'OR' in ARC_NAME:
            columns = list(range(NUM_CLASSES))[:-1]
        elif 'REG' in ARC_NAME:
            columns = ['regression']
        elif NUM_CLASSES > 2:
            columns = list(range(NUM_CLASSES))
        else:
            columns = ['pred_prob']
        pred_classes = {}
        for dset in datasets:
            if 'OR' in ARC_NAME:
                pred_classes[dset] = pd.DataFrame(preds[dset]).apply(special.expit)
                pred_classes[dset] = pred_classes[dset].apply(lambda x: x > 0.5).sum(axis=1)
            elif 'REG' in ARC_NAME:
                pred_classes[dset] = preds[dset]
            elif NUM_CLASSES > 2:
                pred_classes[dset] = np.argmax(preds[dset], axis=-1).astype(np.uint8)
            else:
                pred_classes[dset] = (preds[dset] >= class_thresholds[dset]).astype(np.uint8)
            df = pd.DataFrame(preds[dset], index=input_dict[dset]['clin'].getIDs(), columns=columns)
            df['pred_class'] = pred_classes[dset]
            df['label'] = labels[dset]
            df.to_csv(join(RESULTS_PATH, results_name + "_" + dset + "_pred.csv"))
            print('Saving predictions to:', join(RESULTS_PATH, results_name + "_" + dset + "_pred.csv"))

        tf.keras.backend.clear_session()

    ############################################
    # Get mean and std for metrics
    ############################################

    # Prepare rows with mean and std values.
    result_table['mean'] = {'database': DATABASE, 'date': dt_string}
    result_table['std'] = {'database': DATABASE, 'date': dt_string}
    if 'OR' in ARC_NAME:
        result_table2['mean'] = {'database': DATABASE, 'date': dt_string}
        result_table2['std'] = {'database': DATABASE, 'date': dt_string}

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

    if 'OR' in ARC_NAME:
        metrics_gathered2 = {'train': {}, 'val': {}, 'test': {}}
        for dset in datasets:
            for metric_name in results_method2['train'].keys():
                metrics_gathered2[dset][metric_name] = [result_table2['fold_' + str(f_idx)][metric_name + '_' + dset]
                                                        for f_idx in range(len(folds_list))]
                result_table2['mean'].update(
                    {metric_name + '_' + dset: np.array(metrics_gathered2[dset][metric_name]).mean()})
                result_table2['std'].update(
                    {metric_name + '_' + dset: np.array(metrics_gathered2[dset][metric_name]).std()})

            print('')
            print('OR method 2 ' + dset + ' bACC:', metrics_gathered2[dset]['bacc'])
            print('OR method 2 ' + 'Mean ' + dset + ' bACC: ', result_table2['mean']['bacc_' + dset])
            print('OR method 2 ' + 'STD ' + dset + ' bACC: ', result_table2['std']['bacc_' + dset])

        logging.debug('OR method 2 ' + 'Row mean: %s', result_table2['mean'])
        logging.debug('OR method 2 ' + 'Row std: %s', result_table2['std'])

    ############################################
    # Save to dataframe
    ############################################

    if exists(PERFORMANCE_CSV_PATH):
        df1 = pd.read_csv(PERFORMANCE_CSV_PATH, index_col=0)
        df2 = pd.DataFrame.from_dict(result_table, orient='index')
        df = pd.concat([df1, df2])
        if 'OR' in ARC_NAME:
            df3 = pd.DataFrame.from_dict(result_table2, orient='index')
            df = pd.concat([df, df3])
    else:
        df = pd.DataFrame.from_dict(result_table, orient='index')
        if 'OR' in ARC_NAME:
            df3 = pd.DataFrame.from_dict(result_table2, orient='index')
            df = pd.concat([df, df3])

    df.to_csv(PERFORMANCE_CSV_PATH)
    print('Performance saved to:', PERFORMANCE_CSV_PATH)


# noinspection PyShadowingNames
def prepare_sets(id_lists, fold_i, DG_X_clin, DG_X_dwi, DG_X_flair, DG_y, DG_y_pretrain, models_path):
    ############################################
    # Split into training/ validation / test sets:
    ############################################

    # DG_X_clin_train, DG_X_clin_val, DG_X_clin_test = pipe.splitSet(DG_X_clin, \
    # id_lists['train'], id_lists['val'], id_lists['test'])

    DG_X_clin_train = DG_X_clin.ID_split(id_lists['train'])
    DG_X_clin_val = DG_X_clin.ID_split(id_lists['val'])
    DG_X_clin_test = DG_X_clin.ID_split(id_lists['test'])

    # ID check clinical
    DG_X_flair_train = DG_X_clin_train.IDmatch(DG_X_flair)
    DG_X_flair_val = DG_X_clin_val.IDmatch(DG_X_flair)
    DG_X_flair_test = DG_X_clin_test.IDmatch(DG_X_flair)

    DG_X_dwi_train = DG_X_clin_train.IDmatch(DG_X_dwi)
    DG_X_dwi_val = DG_X_clin_val.IDmatch(DG_X_dwi)
    DG_X_dwi_test = DG_X_clin_test.IDmatch(DG_X_dwi)

    DG_X_flair_train = DG_X_dwi_train.IDmatch(DG_X_flair_train)
    DG_X_flair_val = DG_X_dwi_val.IDmatch(DG_X_flair_val)
    DG_X_flair_test = DG_X_dwi_test.IDmatch(DG_X_flair_test)

    DG_y_train = DG_X_clin_train.IDmatch(DG_y)
    DG_y_val = DG_X_clin_val.IDmatch(DG_y)
    DG_y_test = DG_X_clin_test.IDmatch(DG_y)

    if DG_y_pretrain:
        DG_y_pretrain_train = DG_y_train.IDmatch(DG_y_pretrain)
        DG_y_pretrain_val = DG_y_val.IDmatch(DG_y_pretrain)
        DG_y_pretrain_test = DG_y_test.IDmatch(DG_y_pretrain)

        if not (len(DG_X_clin_train) == len(DG_X_dwi_train) == len(DG_X_flair_train) == len(DG_y_train) ==
                len(DG_y_pretrain_train)):
            print(colored('The inputs do not have the same dimensions, re-apply IDmatch method', 'red'))
            print('len clin:', len(DG_X_clin_train))
            print('len dwi:', len(DG_X_dwi_train))
            print('len flair:', len(DG_X_flair_train))
            print('len y:', len(DG_y_train))
            print('len y pretrain:', len(DG_y_pretrain_train))
    else:
        if not (len(DG_X_clin_train) == len(DG_X_dwi_train) == len(DG_X_flair_train) == len(DG_y_train)):
            print(colored('The inputs do not have the same dimensions, re-apply IDmatch method', 'red'))
            print('len clin:', len(DG_X_clin_train))
            print('len dwi:', len(DG_X_dwi_train))
            print('len flair:', len(DG_X_flair_train))
            print('len y:', len(DG_y_train))

    print('No. of patients in training set:', len(DG_X_clin_train))
    print('No. of patients in validation set:', len(DG_X_clin_val))
    print('No. of patients in test set:', len(DG_X_clin_test))
    if len(DG_X_clin_train) != len(DG_y_train) or len(DG_X_clin_val) != len(DG_y_val) or len(DG_X_clin_test) != len(
            DG_y_test):
        print(colored('Huston, we got a problem! X data dimensions does not fit to y labels dimensions!', 'red'))

    # Save patients in splits
    patients = {'train': list(DG_X_clin_train.getIDs()),
                'val': list(DG_X_clin_val.getIDs()),
                'test': list(DG_X_clin_test.getIDs())}
    with open(join(models_path, 'fold_' + str(fold_i) + 'patient_splits.p'), 'wb') as handle:
        pickle.dump(patients, handle)

    x_dict = {'train':
                  {'clin': DG_X_clin_train,
                   'dwi': DG_X_dwi_train,
                   'flair': DG_X_flair_train},
              'val':
                  {'clin': DG_X_clin_val,
                   'dwi': DG_X_dwi_val,
                   'flair': DG_X_flair_val},
              'test':
                  {'clin': DG_X_clin_test,
                   'dwi': DG_X_dwi_test,
                   'flair': DG_X_flair_test}}
    y_dict = {'train': DG_y_train,
              'val': DG_y_val,
              'test': DG_y_test}

    y_pretrain_dict = {'train': None,
                       'val': None,
                       'test': None}
    if DG_y_pretrain:
        y_pretrain_dict = {'train': DG_y_pretrain_train,
                           'val': DG_y_pretrain_val,
                           'test': DG_y_pretrain_test}

    return x_dict, y_dict, y_pretrain_dict


def predict_sets(set_list, pipe, input_dict, label_dict, model, expand_dims, ae=False):
    pred_probs = {}
    labels = {}

    for dset in set_list:
        features, labels[dset] = pipe.getBatch(
            [input_dict[dset], label_dict[dset]], input_dict[dset]['clin'].getIDs(),
            output_format='np.ndarray', expand_dims=expand_dims)
        features = pipe.preprocess_all_data(features, ['clin', 'dwi', 'flair'])

        pred = model.predict(features)

        # logging.debug(dset + ' PREDICTION SHAPE: %s', pred.shape)
        # logging.debug(dset + ' PREDICTION: %s', pred)

        if ae:
            pred_probs[dset] = copy.deepcopy(pred[0])
        else:
            pred_probs[dset] = copy.deepcopy(pred)

        logging.debug(dset + ' PREDICTION: %s', pred_probs[dset][:10])

    return pred_probs, labels


def parse_commandline_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", help="MODE options: 'dwi', 'flair', 'clin', or combinations e.g. "
                                             "'dwi_flair_clin', 'adc_flair_clin'")
    parser.add_argument("-a", "--arc_name", help="ARC_NAME options: resnet, mobilenet, densenet, groupnet, clinicalNN, "
                                                 "convnet; add 'OR' to the arch name to indicate the ordinal "
                                                 "classification with CORAL, e.g. convnet_OR; add 'REG' to the arch "
                                                 "name to indicate the standard regression, e.g. convnet_REG; add "
                                                 "'COMB_CROSS' to the arch name to indication combined loss of "
                                                 "categorical and binary crossentropy, e.g. convnet_COMB_CROSS; "
                                                 "add 'AE' to the arch name indicate the adding of the decoder path "
                                                 "to the given architecture as in autoencoders, e.g "
                                                 "convnet_AE or convnet_REG_AE")
    parser.add_argument("-s", "--clin_feature_set", help="CLIN_FEATURES_SET options: 'dicho_baseline', "
                                                         "'dicho_baseline_img_ext', 'dicho_release', "
                                                         "'dicho_release_img_ext', 'dicho_shift_baseline', "
                                                         "'dicho_shift_baseline_img_ext', 'fullrange_baseline', "
                                                         "'fullrange_baseline_img_ext'")
    parser.add_argument("--pretrain_label", help="PRETRAIN_LABEL options: None for no pretraining; "
                                                 "options for pretraining: 'LESION_VOL_1D', 'ASPECTS_1D'")
    parser.add_argument("--pretrain_mode", help="PRETRAIN_MODE options: # None for no pretraining; "
                                                "options for pretraining: 'reg' for regression,"
                                                " 'class' for binary classification")
    parser.add_argument("--pretrain_freezing", help="PRETRAIN_FREEZING options: None for no pretraining; "
                                                    "options for pretraining: '0' for tuning dwi model,"
                                                    " '1' for freezing pretrained dwi model")
    parser.add_argument("-f", "--fold", help="crossvalidation fold number(s)")

    args = parser.parse_args()

    mode = args.mode
    arc_name = args.arc_name
    clin_feature_set = args.clin_feature_set
    pretrain_label = args.pretrain_label
    pretrain_mode = args.pretrain_mode
    pretrain_freezing = int(args.pretrain_freezing) if args.pretrain_freezing is not None else None
    if args.fold is not None:
        fold_list = args.fold.split(',')
        fold_numbers = [int(n.strip()) for n in fold_list]
    else:
        fold_numbers = None

    return mode, arc_name, clin_feature_set, pretrain_label, pretrain_mode, pretrain_freezing, fold_numbers


if __name__ == '__main__':
    main()
