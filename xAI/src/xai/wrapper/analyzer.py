'''
Multi-Input Analyzer wrapper.
Runs predefined analyzers.
'''

from typing import List, Tuple, Union
import os
import math
import numpy as np

from xAI.src.xai.utils import data
from xAI.src.xai.utils import heatmap as hm

DWI = 'dwi'
FLAIR = 'flair'
MULTI = 'multi'

class Analyzer():
    '''Wrapper class to computes heatmaps for a list of analyzers.'''

    def __init__(self,
                 analyzers: list,
                 imaging_data: List[List[str]],
                 numeric_data: Union[List[str], None],
                 output_path: str,
                 ndim: int,
                 batch_size: int,
                 preprocess: str = '',
                 hm_standardize: bool = False,
                 hm_normalize: bool = False,
                 hm_threshold: bool = False,
                 hm_convert: bool = False,
                 hm_pad: bool = False,
                 hm_to_grayscale = True):
        '''
        Args:
            analyzers (list): Istantiated analizers.
            imaging_data (List[List[str]]): Paths to nifti images.
            numeric_data (None | List[str]): Paths to numeric data.
            output_path (str): Output directory for the heatmaps to be saved.
            ndim (int): Number of image dimensions (2 for 2D images, 3 for 3D images).
            batch_size (int): Size of a batch of images.
            preprocess (str): Specifies which input to preprocess.
                              Options: `dwi`, `flair`, `multi`.
            hm_normalize (bool): Whether to normalize heatmaps. Defaults to False.
            hm_threshold (bool): Whether to threshold heatmaps. Defaults to False.
            hm_pad (bool): Whether to pad z dim by zeros. Defaults to False.
            hm_to_grayscale (bool): Whether the generated heatmap should be converted to grayscale.
        '''

        self.analyzers = analyzers
        self.imaging_data = imaging_data
        self.numeric_data = numeric_data
        self.batch_size = batch_size
        self.output_path = output_path
        self.ndim = ndim
        self.preprocess = preprocess
        self.hm_standardize = hm_standardize
        self.hm_normalize = hm_normalize
        self.hm_threshold = hm_threshold
        self.hm_convert = hm_convert
        self.hm_pad = hm_pad
        self.hm_to_grayscale = hm_to_grayscale

    def compute_heatmaps(self) -> None:
        '''Computes and saves nifti heatmaps.'''

        print('='*100)
        # get the total number of samples
        num_samples = len(self.imaging_data[0])
        print('Number of samples:', num_samples)
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

        # model specific input preprocessing
        if self.preprocess == DWI:
            inputs[0] = data.preprocess_dwi(inputs[0]) # [0] since single input
        elif self.preprocess == FLAIR:
            inputs[0] = data.preprocess_flair(inputs[0]) # [0] since single input
        elif self.preprocess == MULTI:
            inputs[0] = data.preprocess_dwi(inputs[0])
            inputs[1] = data.preprocess_flair(inputs[1])

        # specific case: single input, single batch
        # saves into the same folder where the input image is
        # self.output_path = os.path.dirname(self.imaging_data[0][start])

        # run analyzers on the batch
        self._run_analyzers(inputs, iaffines, inames)

    def _process_imaging_data(self, start: int, stop: int) -> Tuple:
        '''
        Loads a batch of imaging data.

        Args:
            start (str): Start index.
            stop (str): Stop index.

        Returns:
            Tuple[List[np.ndarray], List[List[np.ndarray]], List[List[str]]]:
                Loaded nifti images, their affine transformations and image file names.
        '''

        # will be List[List[]] for each model's input
        # outer List -> model inputs
        # nested List -> batch number of examples per input
        inputs, iaffines, inames = [], [], []

        for group_paths in self.imaging_data:
            images, affines, names = [], [], []
            batch_paths = group_paths[start:stop]

            # iterate over each path in the batch
            for path in batch_paths:
                # shape [batch, x1, x2, x3, channel]
                image, affine = data.load_nifti_image(path)
                name = os.path.basename(path)

                images.append(image)
                affines.append(affine)
                names.append(name)

            # concatenate over a batch dimension
            inputs.append(np.concatenate(images, axis=0))
            iaffines.append(affines)
            inames.append(names)

        return inputs, iaffines, inames

    def _process_clinical_data(self, start, stop) -> List[np.array]:
        '''
        Loads a batch of clinical data.

        Args:
            start (str): Start index.
            stop (str): Stop index.

        Returns:
            List[np.ndarray]: Loaded numeric data.
                              If no numeric data, returns None.

        '''

        clin_data = []

        batch_paths = self.numeric_data[start:stop]
        for path in batch_paths:
            # shape [batch, x]
            clin_data.append(data.read_clinical_data(path))

        # concatenate over a batch dimension
        clin_data = np.concatenate(clin_data, axis=0)

        return clin_data

    def _run_analyzers(self, inputs: np.ndarray, iaffines: list, inames: list) -> None:
        '''
        Runs defined alayzers on a batch of images.

        Args:
            images (np.ndarray): Batch of images.
            iaffines (list<np.ndarray>): Affines transformations of each image.
            inames (list<str>): Image file names.
        '''
        # `i` as model's input

        for analyzer in self.analyzers:
            print(f'Computing: {analyzer.name}...')

            # `i` as model's input
            ihmaps = analyzer.analyze(inputs)
            if not 'gradcam' in analyzer.name and not isinstance(ihmaps, list):
                ihmaps = [ihmaps]

            if 'gradcam' in analyzer.name:

                input_index = analyzer.input_index

                # heatmap - ndarray of shape (batch_size, H, W D, color_channels)
                # GradCAM produces heatmaps without color dims -> therefore heatmap.ndim-ndim will be 0
                heatmaps = [hm.process_color_channel(heatmap) if heatmap.ndim-self.ndim > 0 else heatmap
                            for heatmap in ihmaps]

                heatmaps = self._postprocess(heatmaps)

                if self.hm_pad:
                    heatmaps = [np.pad(heatmap, [(0, 0), (0, 0), (1, 1)], mode='constant', constant_values=0) for
                                heatmap in heatmaps]

                # save generated heatmaps
                for heatmap, affine, name in zip(heatmaps, iaffines[input_index], inames[input_index]):
                    hm.save(heatmap, affine, self.output_path, analyzer.name, name)
                    print("Heatmap shape:", heatmap.shape)

            else:
                # iterate over each model's imaging input
                for heatmaps, affines, names in zip(ihmaps, iaffines, inames):

                    # process color dimension
                    if self.hm_to_grayscale:
                        heatmaps = [hm.process_color_channel(heatmap) if heatmap.ndim-self.ndim > 0 else heatmap
                                    for heatmap in heatmaps]

                    heatmaps = self._postprocess(heatmaps)

                    if self.hm_pad:
                        if self.hm_to_grayscale:
                            heatmaps = [np.pad(heatmap, [(0, 0), (0, 0), (1, 1)], mode='constant', constant_values=0)
                                        for heatmap in heatmaps]
                        else:
                            heatmaps = [
                                np.pad(heatmap, [(0, 0), (0, 0), (1, 1), (0, 0)], mode='constant', constant_values=0)
                                for heatmap in heatmaps]

                    # save generated heatmaps
                    for heatmap, affine, name in zip(heatmaps, affines, names):
                        hm.save(heatmap, affine, self.output_path, analyzer.name, name,
                                channels=not self.hm_to_grayscale)
                        print("Heatmap shape:", heatmap.shape)

    def _postprocess(self, heatmaps: np.ndarray) -> List[np.ndarray]:
        '''
        Processes computed attribution maps.

        Args:
            heatmaps (np.ndarray): Heatmaps to be processed.

        Returns:
            List[np.ndarray]: List of processed heatmaps.
        '''

        if self.hm_standardize:
            heatmaps = [np.maximum(heatmap, 0) for heatmap in heatmaps]
            heatmaps = [hm.filter_extreme(heatmap) for heatmap in heatmaps]
            heatmaps = [hm.standardize(heatmap) for heatmap in heatmaps]
        if self.hm_normalize:
            heatmaps = [hm.normalize(heatmap) for heatmap in heatmaps]
        if self.hm_threshold:
            heatmaps = [hm.threshold(heatmap) for heatmap in heatmaps]
        if self.hm_convert:
            heatmaps = [hm.convert(heatmap) for heatmap in heatmaps]

        return heatmaps
