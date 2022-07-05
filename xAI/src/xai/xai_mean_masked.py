'''Main module to run analyzers.'''

import os

import numpy as np

from xAI.src.xai.utils import data
from xAI.src.xai.utils.config.parser import Parser
from xAI.src.xai.utils.data import load_nifti_image
import nibabel as nib


#------------------------------------------------------------------------------

def main():
    '''Main.'''

    # params path
    print(os.getcwd())
    params_path = './xAI/src/xai/config_dwi_tti.yml'
    parser = Parser(params_path)

    # get dwi input paths
    dwi_paths = sorted(data.get_images(parser.output_path, '*dwi+flair.*', '.gz'))

    # load images
    all_imgs_list = []
    for i, path in enumerate(dwi_paths):
        img, affine = load_nifti_image(path)
        img = np.squeeze(img)
        if i == 0:
            print('Loaded image shape:', img.shape)
        all_imgs_list.append(img)

    print('Number of loaded images:', len(all_imgs_list))

    # compute heatmap mean
    mean_heatmaps = np.mean(np.array(all_imgs_list), axis=0)
    nib.save(nib.Nifti1Image(mean_heatmaps, affine=np.eye(4)), os.path.join(parser.output_path, 'mean_heatmap.nii.gz'))

    print('Mean heatmap shape:', mean_heatmaps.shape)

    # subtract mean from the individual heatmaps and save
    for i, path in enumerate(dwi_paths):
        img, affine = load_nifti_image(path)
        new_img = np.squeeze(img) - mean_heatmaps

        # clip negative values
        new_img = np.clip(new_img, 0, None)

        if i == 0:
            print('New image shape:', new_img.shape)

        path_parts = path.split('.')
        if path_parts[0] == '':
            output_path = '.' + path_parts[1] + '_mean_masked'
        else:
            output_path = path_parts[0] + '_mean_masked'
        for i, p in enumerate(path_parts):
            if i > 1:
                output_path += '.' + p

        print("Saving to:", output_path)
        nib.save(nib.Nifti1Image(new_img, affine), output_path)


if __name__ == "__main__":
    main()
