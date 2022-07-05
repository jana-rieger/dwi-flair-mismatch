'''
Splits 3D volume into patches, computes heazmaps and then glues them back.

Currently supports only single input and sngle batch.
'''

from typing import List, Tuple, Union, Iterable
import os
import numpy as np
from utils import data
from utils import heatmap as hm
from wrapper.analyzer import Analyzer

class PatchAnalyzer(Analyzer):
    '''Wrapper class to computes patched heatmaps for a list of analyzers.'''

    def __init__(self,
                 analyzers: list,
                 imaging_data: List[List[str]],
                 numeric_data: Union[List[str], None],
                 output_path: str,
                 ndim: int,
                 batch_size: int,
                 preprocess: str = '',
                 patch_size: Iterable = (64, 128),
                 patch_size_z: Iterable = (8, 16),
                 hm_standardize: bool = False,
                 hm_normalize: bool = False,
                 hm_threshold: bool = False,
                 hm_convert: bool = False):
        '''
        Args:
            analyzers (list): Istantiated analizers.
            imaging_data (List[List[str]]): Paths to nifti images.
            numeric_data (None | List[str]): Paths to numeric data.
            output_path (str): Output directory for the hetmaps to be saved.
            ndim (int): Number of image dimensions (2 for 2D images, 3 for 3D images).
            batch_size (int): Size of a batch of images. Must be 1 for this analyzer to work.
            preprocess (str): Specifies which input to preprocess.
                              Options: `dwi`, `flair`, `multi`.
            hm_normalize (bool): Whether to normalize heatmaps. Defaults to True.
            hm_threshold (bool): Whether to threshold heatmaps. Defaults to True.
        '''

        super().__init__(analyzers, imaging_data, numeric_data, output_path, ndim, batch_size,
                         preprocess, hm_standardize, hm_normalize, hm_threshold, hm_convert)

        self.patch_size = patch_size
        self.patch_size_z = patch_size_z

    def _process_batch(self, start: int, stop: int) -> None:
        '''
        Loads images in the paths.

        Args:
            start (str): Start index.
            stop (str): Stop index.
        '''

        inputs, iaffines, inames = self._process_imaging_data(start, stop)

        if self.numeric_data:
            inputs.append(self._process_clinical_data(start, stop))

        # specific case: single input, single batch
        # saves into the same folder where the input image is
        self.output_path = os.path.dirname(self.imaging_data[0][start])

        # run analyzers on the batch
        self._run_analyzers(inputs, iaffines, inames)

    def _run_analyzers(self, inputs: np.ndarray, iaffines: list, inames: list) -> None:
        '''
        Runs defined alayzers on a batch of images.

        Args:
            images (np.ndarray): Batch of images.
            iaffines (list<np.ndarray>): Affines transformations of each image.
            inames (list<str>): Image file names.
        '''

        for analyzer in self.analyzers:
            print(f'Computing: {analyzer.name}...')

            heatmap = self._compute_patched_volume(analyzer, inputs[0], self.patch_size, self.patch_size_z)
            heatmap = np.maximum(heatmap, 0)
            heatmap = hm.filter_extreme(heatmap)
            heatmap = hm.standardize(heatmap)
            heatmap = hm.normalize(heatmap)

            hm.save(heatmap, iaffines[0][0], self.output_path, analyzer.name, inames[0][0])

    def _compute_patched_volume(self, analyzer, volume: np.ndarray, patch_size: Iterable, patch_size_z: Iterable):
        '''
        TODO
        '''

        # final_hmap -> prob_mat
        # img_mat_temp -> volume
        # patch_size=[64, 128], patch_size_z=[8,16]

        volume = np.squeeze(volume)

        # final heatmap
        final_hmap = np.zeros(volume.shape, dtype=np.float32)

        x_min = y_min = z_min = 0
        x_max, y_max, z_max = volume.shape

        num_x_patches = np.int(np.ceil((x_max - x_min) / patch_size[0]))
        num_y_patches = np.int(np.ceil((y_max - y_min) / patch_size[0]))	
        num_z_patches = np.int(np.ceil((z_max - z_min) / patch_size_z[0]))

        if num_z_patches*patch_size_z[0] + (np.max(patch_size_z)-np.min(patch_size_z))//2 > volume.shape[2]:
            new_z = (num_z_patches-1)*patch_size_z[0] + patch_size_z[0]//2 + np.max(patch_size_z)//2 # so that we can feed sufficient patches
            temp = np.zeros((volume.shape[0], volume.shape[1], new_z))
            temp[:, :, :volume.shape[2]] = volume
            temp[:, :, volume.shape[2]:] = volume[:, :, -(new_z - volume.shape[2]):]
            volume = temp

        for ix in range(num_x_patches):
            for iy in range(num_y_patches):
                for iz in range(num_z_patches):
                    # find the starting and ending x and y coordinates of given patch
                    patch_start_x = patch_size[0] * ix
                    patch_end_x = patch_size[0] * (ix + 1)
                    patch_start_y = patch_size[0] * iy
                    patch_end_y = patch_size[0] * (iy + 1)
                    patch_start_z = patch_size_z[0] * iz
                    patch_end_z = patch_size_z[0] * (iz + 1)
                    if patch_end_x > x_max:
                        patch_end_x = x_max
                    if patch_end_y > y_max:
                        patch_end_y = y_max
                    if patch_end_z > z_max:
                        patch_end_z = z_max

                    # find center loc with ref. size
                    center_x = patch_start_x + int(patch_size[0]/2)
                    center_y = patch_start_y + int(patch_size[0]/2)
                    center_z = patch_start_z + int(patch_size_z[0]/2)

                    img_patches = []
                    for size, size_z in zip(patch_size, patch_size_z):
                        img_patch = np.zeros((size, size, size_z, 1))
                        offset_x = 0
                        offset_y = 0
                        offset_z = 0

                        # find the starting and ending x and y coordinates of given patch
                        img_patch_start_x = center_x - int(size/2)
                        img_patch_end_x = center_x + int(size/2)
                        img_patch_start_y = center_y - int(size/2)
                        img_patch_end_y = center_y + int(size/2)
                        img_patch_start_z = center_z - int(size_z/2)
                        img_patch_end_z = center_z + int(size_z/2)

                        if img_patch_end_x > x_max:
                            img_patch_end_x = x_max
                        if img_patch_end_y > y_max:
                            img_patch_end_y = y_max
                        if img_patch_start_x < x_min:
                            offset_x = x_min - img_patch_start_x
                            img_patch_start_x = x_min
                        if img_patch_start_y < y_min:
                            offset_y = y_min - img_patch_start_y
                            img_patch_start_y = y_min
                        if img_patch_start_z < z_min:
                            offset_z = z_min - img_patch_start_z
                            img_patch_start_z = z_min

                        # get the patch with the found coordinates from the image matrix
                        img_patch[offset_x : offset_x + (img_patch_end_x-img_patch_start_x),
                                  offset_y : offset_y + (img_patch_end_y-img_patch_start_y),
                                  offset_z : offset_z + (img_patch_end_z-img_patch_start_z), 0] = \
                           volume[img_patch_start_x: img_patch_end_x,
                                  img_patch_start_y: img_patch_end_y,
                                  img_patch_start_z:img_patch_end_z]

                        img_patches.append(np.expand_dims(img_patch.astype(np.float32), 0))

                    # compute heatmap
                    ihmaps = analyzer.analyze(img_patches)
                    if isinstance(ihmaps, list):
                        hmap = ihmaps[0]
                    else:
                        hmap = ihmaps

                    # hmap: ndarray of shape (batch_dim, H, W, D, color_dim)
                    # batch_dim is alwyas 1
                    hmap = np.squeeze(hmap)
                    #hmap = hm.normalize(hmap)

                    final_hmap[patch_start_x : patch_end_x,
                               patch_start_y : patch_end_y,
                               patch_start_z : patch_end_z] = \
                        np.reshape(hmap, (patch_size[0], patch_size[0], patch_size_z[0])) \
                              [:patch_end_x-patch_start_x,
                               :patch_end_y-patch_start_y,
                               :patch_end_z-patch_start_z]

        return final_hmap
