'''
Multi-Input Analyzer wrapper for npz input files.
Runs predefined analyzers.
'''

from typing import List, Tuple, Union
import os
import math
import numpy as np

from utils import data
from utils import heatmap as hm

class Analyzer():
    '''Wrapper class to computes heatmaps for a list of analyzers'''

    def __init__(self,
                 analyzers: list,
                 imaging_data: List[List[str]],
                 numeric_data: Union[List[str], None],
                 output_path: str,
                 ndim: int,
                 batch_size: int,
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
            batch_size (int): Size of a batch of images.
            hm_normalize (bool): Whether to normalize heatmaps. Defaults to True.
            hm_threshold (bool): Whether to threshold heatmaps. Defaults to True.
        '''

        self.analyzers = analyzers
        self.imaging_data = imaging_data
        self.numeric_data = numeric_data
        self.batch_size = batch_size
        self.output_path = output_path
        self.ndim = ndim
        self.hm_standardize = hm_standardize
        self.hm_normalize = hm_normalize
        self.hm_threshold = hm_threshold
        self.hm_convert = hm_convert

    def compute_heatmaps(self) -> None:
        '''Computes and saves nifti heatmaps.'''

        print('='*100)
        # get the total number of samples
        num_samples = len(self.imaging_data[0])
        # number of full batch steps
        steps = math.floor(num_samples/self.batch_size)
        for i in range(steps):
            print(f'Running batch #{i+1}...')

            # extract a batch [i*size:(i+1)*size]
            start_index = i*self.batch_size
            stop_index = (i+1)*self.batch_size
            self._process_batch(start_index, stop_index)

        # if non-interger number of batches
        left_samples = num_samples % self.batch_size
        if left_samples > 0:
            print(f'Running batch #{steps+1}...')

            # extract a batch
            start_index = steps*self.batch_size
            stop_index = steps*self.batch_size + 1
            self._process_batch(start_index, stop_index)

        print('='*100)

    def _process_batch(self, start: int, stop: int) -> tuple:
        '''
        Loads images in the paths.

        Args:
            start (str): Start index.
            stop (str): Stop index.

        Returns:
            Tuple[List[np.ndarray], List[List[np.ndarray]], List[List[str]]]: Loaded nifti images, their affine
                                                             transformations and image file names.
        '''

        inputs, inames = self._process_imaging_data(start, stop)

        # run analyzers on the batch
        self._run_analyzers(inputs, inames)

    def _process_imaging_data(self, start: int, stop: int) -> Tuple:
        '''
        Loads a batch of imaging data.

        Args:
            start (str): Start index.
            stop (str): Stop index.
        '''

        # will be List[List[]] for each model's input
        # 'i' stands for model's 'input'
        inputs, inames = [], []

        for group_paths in self.imaging_data:
            images, names = [], []
            batch_paths = group_paths[start:stop]

            # iterate over each path in the batch
            for path in batch_paths:
                # shape [batch, x1, x2, x3, channel]
                image = data.load_npz_image(path)
                name = os.path.basename(path)

                images.append(image)
                names.append(name)

            # concatenate over a batch dimension
            inputs.append(np.concatenate(images, axis=0))
            inames.append(names)

        return inputs, inames

    def _run_analyzers(self, inputs: np.ndarray, inames: list) -> None:
        '''
        Runs defined alayzers on a batch of images.

        Args:
            images (np.ndarray): Batch of images.
            iaffines (list<np.ndarray>): Affines transformations of each image.
            inames (list<str>): Image file names.
        '''

        for analyzer in self.analyzers:
            print(f'Computing: {analyzer.name}...')

            # `i` as model's input
            ihmaps = analyzer.analyze(inputs)
            if not isinstance(ihmaps, list):
                ihmaps = [ihmaps]

            if 'gradcam' in analyzer.name:

                input_index = analyzer.input_index

                # ihmaps - ndarray of shape (batch_size, H, W D, color_channels)
                heatmaps = [hm.process_color_channel(heatmap) if heatmap.ndim-self.ndim > 0 else heatmap
                            for heatmap in ihmaps]

                heatmaps = self._postprocess(heatmaps)

                # save generated heatmaps
                for heatmap, name in zip(heatmaps, inames[input_index]):
                    hm.save_npz(heatmap, self.output_path, analyzer.name, name)

            else:
                # iterate over each model's imaging input
                for heatmaps, names in zip(ihmaps, inames):

                    # process color dimension
                    # GradCAM produces heatmaps without color dims ->
                    # therefore heatmap.ndim-ndim will be 0
                    heatmaps = [hm.process_color_channel(heatmap) if heatmap.ndim-self.ndim > 0 else heatmap
                                for heatmap in heatmaps]

                    heatmaps = self._postprocess(heatmaps)

                   # save generated heatmaps
                    for heatmap, name in zip(heatmaps, names):
                        hm.save_npz(heatmap, self.output_path, analyzer.name, name)

    def _postprocess(self, heatmaps: np.ndarray) -> List[np.ndarray]:
        '''
        Processes computed attribution maps.

        Args:
            heatmaps (np.ndarray): Heatmaps to be processed.

        Returns:
            List[np.ndarray]: List of processed heatmaps.
        '''

        if self.hm_standardize:
            heatmaps = [hm.standardize(heatmap) for heatmap in heatmaps]
        if self.hm_normalize:
            heatmaps = [hm.normalize(heatmap) for heatmap in heatmaps]
        if self.hm_threshold:
            heatmaps = [hm.threshold(heatmap) for heatmap in heatmaps]
        if self.hm_convert:
            heatmaps = [hm.convert(heatmap) for heatmap in heatmaps]

        return heatmaps
