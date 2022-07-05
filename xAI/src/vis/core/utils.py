''' Visualization utils.'''

from typing import List, Tuple
import os
import glob
import nibabel as nib
import matplotlib.pyplot

def load_data(impath: str, hmpath: str) -> Tuple:
    '''
    TODO
    '''

    # load the mage
    image = nib.load(impath).get_fdata()

    # get heatmap paths for the image being investigated
    # each heatmap will match with the image by '*[image_name]'
    imname = os.path.basename(impath)
    hmpaths = glob.glob(os.path.join(hmpath, '*' + imname))
    # load heatmaps
    heatmaps = [nib.load(hmpath).get_fdata() for hmpath in hmpaths]

    # get titles
    titles = [os.path.basename(hmpath).split('_')[0] for hmpath in hmpaths]

    if all(['gradcam' in hmpath for hmpath in hmpaths]):
        titles = [os.path.basename(hmpath).split('_')[0] + '_' + os.path.basename(hmpath).split('_')[1]
                  for hmpath in hmpaths]


    return image, heatmaps, titles

def remove_keymap_conflicts(new_keys_set: set, plt: matplotlib.pyplot):
    '''
    Removes given keys from the key mappings to avoid conflicts.

    Args:
        new_keys_set (set):
    '''

    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def get_titles(hmpaths: List[str]) -> List[str]:
    '''
    Extracts analyzer's name from the heatmap names.

    Args:
        hmpaths (list[str]): List of heatmap names.

    Returns:
        list[str]: List of analyzer names.
    '''

    titles = [os.path.basename(hmpath).split('_')[0] for hmpath in hmpaths]

    return titles

def interpolate(colormap, x):
    x = max(0.0, min(1.0, x))
    a = int(x*255.0)
    b = min(255, a + 1)
    f = x*255.0 - a
    return [colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f,
            colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f,
            colormap[a][2] + (colormap[b][2] - colormap[a][2]) * f]
