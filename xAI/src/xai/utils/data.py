'''Data utilities.'''

from typing import List, Set

import os
import glob
import re
import csv
import nibabel as nib
import numpy as np


def get_images(data_path: str, name_pattern: str, ext: str) -> List[str]:
    '''
    Gets full paths of nifti images.

    Args:
        data_path (str): Path to nifti images.
        imtag (str): Image tag: `dwi` or `flair`.
        ext (str): Image extension: `.nii` or `.gz`.

    Returns:
        List[str]: Image paths.
    '''

    # define search pattern
    pattern = name_pattern + ext
    # get full paths
    images = glob.glob(os.path.join(data_path, pattern))

    return images


def filter_img_paths(img_paths, filter_list):
    filtered = [path for path in img_paths if os.path.basename(path)[6:10] in filter_list]
    print('Number of filtered images:', len(filtered))

    return filtered


def get_patient_ids_from_test_fold(model_path, folds_path):
    fold_idx = int(os.path.basename(model_path)[4])
    print('Fold idx:', fold_idx)

    folds = np.load(folds_path, allow_pickle=True)
    test_set = np.sort(folds[fold_idx]['test'])

    return test_set


def get_patient_ids_from_csv(csv_path):
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        patients_ids = list(csv_reader)
        
    return patients_ids


def get_clinical_data(data_path: str, image_paths: List[str]) -> List[str]:
    '''
    Gets fulls paths of clinical data.

    Args:
        data_path (str): Path to nifti images.
        image_paths (List[str]): Image paths.

    Returns:
        List[str]: Clinical data paths.
    '''

    def contains_all(string: str, elements: Set[str]) -> bool:
        '''
        Checks if all the elements are contained in the sting.

        Args:
            string (str): String to be checked.
            elements (Set[str]): String elements that must be contained in the string.

        Returns:
            bool: True if all the elements are contained in the string,
                  False otherwise.
        '''

        return 0 not in [element in string for element in elements]

    # match clinical data with image paths
    regex = re.compile(r'[\d]{4}')
    file_numbers = [regex.search(path).group() for path in image_paths]
    clinical_paths = glob.glob(os.path.join(data_path, '*'))

    filtered_clinical_paths = [clin_path for number in file_numbers
                               for clin_path in clinical_paths
                               if contains_all(clin_path, {number, 'X'})]

    return filtered_clinical_paths

def are_imaging_data_match(dwi_paths: List[str], flair_paths: List[str]) -> bool:
    '''
    Checks if both dwi and flair data match.

    Args:
        dwi_paths (List[str]): DWI imaging paths.
        flair_paths (List[str]): FLAIR imaging paths.
    '''

    # get number ids from the image names
    regex = re.compile(r'[\d]{4}')
    file_numbers = [regex.search(path).group() for path in dwi_paths]

    result = all([number in path for number, path in zip(file_numbers, flair_paths)])

    return result

def read_clinical_data(clinical_path: str) -> np.ndarray:
    '''
    Reads clinical data from a csv file.

    Args:
        clinical_path (str): Path to a file.

    Returns:
        np.ndarray: Clinical data with an added batch dimension
    '''

    with open(clinical_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = list(csv_reader)

        # [1]: to skip header line
        values = [float(value) for value in data[1]]
        # convert to a numpy array and add a batch dimension
        values = np.expand_dims(np.array(values), axis=0)

        return values

def load_nifti_image(image: str) -> tuple:
    '''
    Loads a nifti image.

    Args:
        image (str): Path to a nifti image.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of numpy image with added batch and color dim
                                       and nifti's affine transformations.
    '''

    # load nifti image
    nii_image = nib.load(image)
    # add batch and channel dimensions [batch, x1, x2, x3, channel]
    np_image = np.expand_dims(nii_image.get_fdata(), axis=0)
    if len(np_image.shape) == 4:
        np_image = np.expand_dims(np_image, -1)
    # extract affine transformations
    nii_affine = nii_image.affine

    return np_image, nii_affine

def preprocess_dwi(image: np.ndarray) -> np.ndarray:
    '''
    Standardizes a dwi image

    Args:
        image (np.ndarray): Image to be standardized.

    Returns:
        np.ndarray: Standardized image.
    '''

    mean = 21.589395510246952
    std = 40.62556713196244

    image = (image - mean) / std

    return image

def preprocess_flair(image: np.ndarray) -> np.ndarray:
    '''
    Standardizes a flair image

    Args:
        image (np.ndarray): Image to be standardized.

    Returns:
        np.ndarray: Standardized image.
    '''

    mean = 85.40128464595077
    std = 106.93693761039843

    image = (image - mean) / std

    return image

def load_npz_image(image: str) -> np.ndarray:
    '''
    Loads a nifti image.

    Args:
        image (str): Path to a nifti image.

    Returns:
        np.ndarray: numpy image with added batch and color dim.
    '''

    # load nifti image
    np_image = np.load(image)
    # add batch and channel dimensions [batch, x1, x2, x3, channel]
    np_image = np.expand_dims(np.expand_dims(np_image['arr_0'], axis=0), -1)

    return np_image
