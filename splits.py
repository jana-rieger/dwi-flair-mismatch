"""
Create cross-validation splits and save as .npy
"""

import os
import numpy as np
import random
from ADT.PiplineADT import Pipeline

random.seed(74)
np.random.seed(32)  # 112, 32

folds = 4

CLIN_FEATURES_SET = 'dicho_tti'  # 'fullrange_baseline'

save_path = "./xval_folds_tti.npy"

# LOCAL
DUF_clin_path = 'data/1000plus_coregistered/clinical/' + CLIN_FEATURES_SET
DUF_img_path = 'data/1000plus_coregistered/imgs'

DUF_clin_regex = '1000plus_clinData_0cov_w_header_X*'  # '1000plus_clinData_11cov_w_header_X*'
DUF_y_regex = '1000plus_clinData_0cov_w_header_y*'  # '1000plus_clinData_11cov_w_header_y*'
DUF_dwi_regex = '1kplus*_dwi.nii.gz'
DUF_flair_regex = '1kplus*_flair.nii.gz'

ID_cut_clin = [-8, -4]
ID_cut_dwi = [6, 10]
ID_cut_flair = [6, 10]

ftype_clin = 'csv_w_header'
ftype_img = 'nii.gz'

pipe = Pipeline({})

# Create a DataGenerator object storing all of the DUF files
DG_X_clin = pipe.createDG(DUF_clin_regex, DUF_clin_path, ID_cut_clin, ftype_clin)
DG_X_dwi = pipe.createDG(DUF_dwi_regex, DUF_img_path, ID_cut_dwi, ftype_img)
DG_X_flair = pipe.createDG(DUF_flair_regex, DUF_img_path, ID_cut_flair, ftype_img)
DG_y = pipe.createDG(DUF_y_regex, DUF_clin_path, ID_cut_clin, ftype_clin)

# match dwi flair y with clin
DG_X_dwi = DG_X_clin.IDmatch(DG_X_dwi)
DG_X_flair = DG_X_clin.IDmatch(DG_X_flair)
DG_X_flair = DG_X_dwi.IDmatch(DG_X_flair)
DG_y = DG_X_clin.IDmatch(DG_y)

# get patients list
patients = DG_X_dwi.getIDs()
print('Number of patients:', len(patients))

# shuffle patients list
random.shuffle(patients)
print(patients)

# Set dataset ratios
training_ratio = 1.0 - (1.0/folds)
val_ratio = 0.25		# selected randomly after train/test split within fold
test_ratio = 1.0/folds

# Define dataset list
folds_list = []

offset = 0
point_per_fold = int(np.floor(len(patients)*test_ratio))
print(point_per_fold)

test_folds = []
for fold in range(folds):
    test_folds.append(patients[offset: offset + point_per_fold])
    offset += point_per_fold

# add residual patients to the last fold
test_folds[-1].append(patients[offset:][0])

for fold in test_folds:
    print(fold)

for current_fold in range(folds):
    fold = {}

    train_val_patients = np.concatenate([fold for i, fold in enumerate(test_folds) if i != current_fold])
    val_split = int(np.floor(len(train_val_patients)*val_ratio))

    np.random.shuffle(train_val_patients)
    fold['train'] = train_val_patients[val_split:]
    fold['val'] = train_val_patients[:val_split]
    fold['test'] = np.array(test_folds[current_fold])

    folds_list.append(fold)

# Saving
print('Saving lists to directory: ', save_path)
with open(save_path, 'wb') as file:
    np.save(file, folds_list, allow_pickle=True)

# Check by loading
print('Loading from directory: ', save_path)
with open(save_path, 'rb') as file:
    folds_list2 = np.load(file, allow_pickle=True)

for fold in folds_list2:
    print(fold)

for fold in folds_list2:
    print(len(fold['train']), len(fold['val']), len(fold['test']),
          'total:', len(fold['train']) + len(fold['val']) + len(fold['test']))
