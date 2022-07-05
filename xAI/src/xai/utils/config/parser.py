'''Parameters parser.'''

from typing import Callable, TypeVar
import yaml
import tensorflow as tf

import xAI.src.xai.utils.config as c
from xAI.src.xai.utils import losses
from xAI.src.xai.analyzers.gradcampp import GradCAMPP
from xAI.src.xai.analyzers.gradcam import GradCAM
from xAI.src.xai.analyzers.vanilla_gradients import VanillaGradients
from xAI.src.xai.analyzers.smoothgrad import SmoothGrad
from xAI.src.xai.analyzers.integrated_gradients import IntegratedGradients
from xAI.src.xai.analyzers.input_gradient import InputGradient
from xAI.src.xai.analyzers.occlusion_sensitivity import OcclusionSensitivity

class Parser():
    '''Parses parameters and instantiates analyzers.'''

    def __init__(self, params_path: str):
        '''
        Args:
            params_path (str): Path to the parameters file.
        '''

        # read the contents of the yaml file
        self.params = read_yaml(params_path)
        # set loss function for analyzers
        self.loss = self._get_loss()
        # num of dims for GradCAM and analyzer wrapper
        self.ndim = self.params[c.CNFG_DATA][c.CNFG_NUM_DIM]

        # for public use:
        self.model_path = self.params[c.CNFG_MODEL][c.CNFG_PATH]
        self.imaging_input = self.params[c.CNFG_DATA][c.CNFG_INPUT_PATH]
        self.clinical_input = self.params[c.CNFG_DATA][c.CNFG_CLINICAL][c.CNFG_INPUT_PATH] \
                                if (c.CNFG_CLINICAL in self.params[c.CNFG_DATA]) else None
        self.folds_path = self.params[c.CNFG_DATA][c.CNFG_FOLDS_PATH]
        self.patient_ids_path = self.params[c.CNFG_DATA][c.CNFG_PATIENT_IDS_PATH]
        self.output_path = self.params[c.CNFG_DATA][c.CNFG_OUTPUT_PATH]

    def parse_analyzer_wrapper(self) -> dict:
        '''Parses and returns analyzer wrapper parameters.'''

        # define required parameters
        params = {}
        params[c.WRAPPER_BATCH_SIZE] = self.params[c.CNFG_DATA][c.CNFG_BATCH_SIZE]
        params[c.WRAPPER_OUTPUT_PATH] = self.params[c.CNFG_DATA][c.CNFG_OUTPUT_PATH]
        params[c.WRAPPER_NDIM] = self.params[c.CNFG_DATA][c.CNFG_NUM_DIM]

        if c.CNFG_HEATMAP in self.params:
            if c.CNFG_STANDARDIZE in self.params[c.CNFG_HEATMAP]:
                params[c.WRAPPER_HM_STANDARDIZE] = self.params[c.CNFG_HEATMAP][c.CNFG_STANDARDIZE]
            if c.CNFG_NORMALIZE in self.params[c.CNFG_HEATMAP]:
                params[c.WRAPPER_HM_NORMALIZE] = self.params[c.CNFG_HEATMAP][c.CNFG_NORMALIZE]
            if c.CNFG_THRESHOLD in self.params[c.CNFG_HEATMAP]:
                params[c.WRAPPER_HM_THRESHOLD] = self.params[c.CNFG_HEATMAP][c.CNFG_THRESHOLD]
            if c.CNFG_CONVERT in self.params[c.CNFG_HEATMAP]:
                params[c.WRAPPER_HM_CONVERT] = self.params[c.CNFG_HEATMAP][c.CNFG_CONVERT]
            if c.CNFG_PAD in self.params[c.CNFG_HEATMAP]:
                params[c.WRAPPER_HM_PAD] = self.params[c.CNFG_HEATMAP][c.CNFG_PAD]
            if c.CNFG_TO_GRAYSCALE in self.params[c.CNFG_HEATMAP]:
                params[c.WRAPPER_HM_TO_GRAYSCALE] = self.params[c.CNFG_HEATMAP][c.CNFG_TO_GRAYSCALE]

        # num of dims for GradCAM and analyzer wrapper
        self.ndim = params[c.WRAPPER_NDIM]

        return params

    def parse_analyzers(self, model: tf.keras.Model) -> list:
        '''
        Reads and parses yaml parameters file.

        Args:
            model (tf.keras.Model): Keras model.

        Returns:
            list: Instantiated analyzers.
        '''

        analyzers = []
        for analyzer in self.params[c.CNFG_ANALYZERS]:
            # has no arguments -> default parameters
            if isinstance(analyzer, str):
                analyzers.append(self._get_default_analyzer(analyzer, model))
            # custom arguments
            elif isinstance(analyzer, dict):
                analyzers.append(self._get_custom_analyzer(analyzer, model))
            else: # error
                return None

        return analyzers

    def _get_loss(self) -> Callable:
        '''Returns a loss function'''

        if self.params[c.CNFG_LOSS_FUNCTION] == c.LOSS_MEAN:
            return losses.mean_loss
        if self.params[c.CNFG_LOSS_FUNCTION] == c.LOSS_MAX:
            return losses.max_loss
        if self.params[c.CNFG_LOSS_FUNCTION] == c.LOSS_IDENTITY:
            return losses.identity_loss

        return None # error in the yaml file

    def _get_default_analyzer(self, name: str, model: tf.keras.Model) -> TypeVar:
        '''
        Instantiates a default analyzer.

        Args:
            name (str): Analyzer.
            model (tf.keras.Model): Keras model.

        Returns:
            TypeVar: Instantiated analyzer.
        '''

        if name == c.CNFG_VANILLA_GRADIENTS:
            return VanillaGradients(model, loss_func=self.loss)
        if name == c.CNFG_SMOOTHGRAD:
            return SmoothGrad(model, loss_func=self.loss)
        # if name == c.CNFG_GUIDED_GRADIENTS:
        #     return GuidedGradients(model, loss_func=self.loss)
        if name == c.CNFG_INTEGRATED_GRADIENTS:
            return IntegratedGradients(model, loss_func=self.loss)
        if name == c.CNFG_INPUTS_GRADIENTS:
            return InputGradient(model, loss_func=self.loss)
        if name == c.CNFG_OCCLUSION_SENSITIVITY:
            return OcclusionSensitivity(model, loss_func=self.loss)
        if name == c.CNFG_GRADCAM:
            return GradCAM(model, loss_func=self.loss, ndim=self.ndim+c.XDIM)
        if name == c.CNFG_GRADCAMPP:
            return GradCAMPP(model, loss_func=self.loss, ndim=self.ndim+c.XDIM)

        return None # error in the yaml file

    def _get_custom_analyzer(self, analyzer: dict, model: tf.keras.Model) -> TypeVar:
        '''
        Reads custom parameters and instantiates corresponding analyzers.

        Args:
            analyzer (dict): Analyzer's custom parameters.
            model (tf.keras.Model): Keras model.

        Returns:
            TypeVar: Instantiated analyzer.
        '''

        if c.CNFG_VANILLA_GRADIENTS in analyzer:
            params = {}
            # will always be true since there's only one optional argument
            if c.CNFG_TAG in analyzer[c.CNFG_VANILLA_GRADIENTS]:
                params[c.ATAG] = analyzer[c.CNFG_VANILLA_GRADIENTS][c.CNFG_TAG]
            return VanillaGradients(model, loss_func=self.loss, **params)

        if c.CNFG_SMOOTHGRAD in analyzer:
            params = {}
            if c.CNFG_NUM_SAMPLES in analyzer[c.CNFG_SMOOTHGRAD]:
                params[c.SMOOTHGRAD_NUM_SAMPLES] = analyzer[c.CNFG_SMOOTHGRAD][c.CNFG_NUM_SAMPLES]
            if c.CNFG_NOISE in analyzer[c.CNFG_SMOOTHGRAD]:
                params[c.SMOOTHGRAD_NOISE] = analyzer[c.CNFG_SMOOTHGRAD][c.CNFG_NOISE]
            if c.CNFG_TAG in analyzer[c.CNFG_SMOOTHGRAD]:
                params[c.ATAG] = analyzer[c.CNFG_SMOOTHGRAD][c.CNFG_TAG]
            return SmoothGrad(model, loss_func=self.loss, **params)

        # if c.CNFG_GUIDED_GRADIENTS in analyzer:
        #     params = {}
        #     # will always be true since there's only one optional argument
        #     if c.CNFG_TAG in analyzer[c.CNFG_GUIDED_GRADIENTS]:
        #         params[c.ATAG] = analyzer[c.CNFG_GUIDED_GRADIENTS][c.CNFG_TAG]
        #     return GuidedGradients(model, loss_func=self.loss, **params)

        if c.CNFG_INTEGRATED_GRADIENTS in analyzer:
            params = {}
            if c.CNFG_STEPS in analyzer[c.CNFG_INTEGRATED_GRADIENTS]:
                params[c.INTGRADS_STEPS] = analyzer[c.CNFG_INTEGRATED_GRADIENTS][c.CNFG_STEPS]
            if c.CNFG_TAG in analyzer[c.CNFG_INTEGRATED_GRADIENTS]:
                params[c.ATAG] = analyzer[c.CNFG_INTEGRATED_GRADIENTS][c.CNFG_TAG]
            return IntegratedGradients(model, loss_func=self.loss, **params)

        if c.CNFG_INPUTS_GRADIENTS in analyzer:
            params = {}
            # will always be true since there's only one optional argument
            if c.CNFG_TAG in analyzer[c.CNFG_INPUTS_GRADIENTS]:
                params[c.ATAG] = analyzer[c.CNFG_INPUTS_GRADIENTS][c.CNFG_TAG]
            return InputGradient(model, loss_func=self.loss, **params)

        if c.CNFG_OCCLUSION_SENSITIVITY in analyzer:
            params = {}
            if c.CNFG_OCCLUDING_VORTEX in analyzer[c.CNFG_OCCLUSION_SENSITIVITY]:
                params[c.OCCLSENS_OCCLUDING_VORTEX] = analyzer[c.CNFG_OCCLUSION_SENSITIVITY][c.CNFG_OCCLUDING_VORTEX]
            if c.CNFG_PATCH_SIZE in analyzer[c.CNFG_OCCLUSION_SENSITIVITY]:
                params[c.OCCLSENS_PATCH_SIZE] = analyzer[c.CNFG_OCCLUSION_SENSITIVITY][c.CNFG_PATCH_SIZE]
            if c.CNFG_TAG in analyzer[c.CNFG_OCCLUSION_SENSITIVITY]:
                params[c.ATAG] = analyzer[c.CNFG_OCCLUSION_SENSITIVITY][c.CNFG_TAG]
            return OcclusionSensitivity(model, loss_func=self.loss, **params)

        if c.CNFG_GRADCAM in analyzer:
            params = {}
            if c.CNFG_INPUT_INDEX in analyzer[c.CNFG_GRADCAM]:
                params[c.GRADCAM_INIDX] = analyzer[c.CNFG_GRADCAM][c.CNFG_INPUT_INDEX]
            if c.CNFG_LAYER_NAME in analyzer[c.CNFG_GRADCAM]:
                params[c.GRADCAM_LAYER_NAME] = analyzer[c.CNFG_GRADCAM][c.CNFG_LAYER_NAME]
            if c.CNFG_GUIDED_GRADS in analyzer[c.CNFG_GRADCAM]:
                params[c.GRADCAM_GUIDED_GRADS] = analyzer[c.CNFG_GRADCAM][c.CNFG_GUIDED_GRADS]
            if c.CNFG_NORM_GRADS in analyzer[c.CNFG_GRADCAM]:
                params[c.GRADCAM_NORM_GRADS] = analyzer[c.CNFG_GRADCAM][c.CNFG_NORM_GRADS]
            if c.CNFG_FILTER_VALUES in analyzer[c.CNFG_GRADCAM]:
                params[c.GRADCAM_FILTER_VALUES] = analyzer[c.CNFG_GRADCAM][c.CNFG_FILTER_VALUES]
            if c.CNFG_TAG in analyzer[c.CNFG_GRADCAM]:
                params[c.ATAG] = analyzer[c.CNFG_GRADCAM][c.CNFG_TAG]
            return GradCAM(model, loss_func=self.loss, ndim=self.ndim+c.XDIM, **params)

        if c.CNFG_GRADCAMPP in analyzer:
            params = {}
            if c.CNFG_INPUT_INDEX in analyzer[c.CNFG_GRADCAMPP]:
                params[c.GRADCAM_INIDX] = analyzer[c.CNFG_GRADCAMPP][c.CNFG_INPUT_INDEX]
            if c.CNFG_LAYER_NAME in analyzer[c.CNFG_GRADCAMPP]:
                params[c.GRADCAM_LAYER_NAME] = analyzer[c.CNFG_GRADCAMPP][c.CNFG_LAYER_NAME]
            if c.CNFG_GUIDED_GRADS in analyzer[c.CNFG_GRADCAMPP]:
                params[c.GRADCAM_GUIDED_GRADS] = analyzer[c.CNFG_GRADCAMPP][c.CNFG_GUIDED_GRADS]
            if c.CNFG_NORM_GRADS in analyzer[c.CNFG_GRADCAMPP]:
                params[c.GRADCAM_NORM_GRADS] = analyzer[c.CNFG_GRADCAMPP][c.CNFG_NORM_GRADS]
            if c.CNFG_FILTER_VALUES in analyzer[c.CNFG_GRADCAMPP]:
                params[c.GRADCAM_FILTER_VALUES] = analyzer[c.CNFG_GRADCAMPP][c.CNFG_FILTER_VALUES]
            if c.CNFG_TAG in analyzer[c.CNFG_GRADCAMPP]:
                params[c.ATAG] = analyzer[c.CNFG_GRADCAMPP][c.CNFG_TAG]
            return GradCAMPP(model, loss_func=self.loss, ndim=self.ndim+c.XDIM, **params)

        return None # error in the yaml file

def read_yaml(file: str) -> dict:
    '''
    Reads a yaml file.

    Args:
        file (str): Path to the yaml file.

    Returns:
        dict: Contents of the yaml file.
    '''

    with open(file, 'r') as yaml_file:
        contents = yaml.full_load(yaml_file)

    return contents
