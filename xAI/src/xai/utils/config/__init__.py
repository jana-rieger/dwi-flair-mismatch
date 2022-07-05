'''Contains constants definitions for config parser module.'''

# config constants

CNFG_DATA = 'data'
CNFG_CLINICAL = 'clinical'
CNFG_MODEL = 'model'
CNFG_LOSS_FUNCTION = 'lossFunction'
CNFG_HEATMAP = 'heatmap'

CNFG_ANALYZERS = 'analyzers'
CNFG_GRADCAM = 'gradcam'
CNFG_GRADCAMPP = 'gradcampp'
CNFG_VANILLA_GRADIENTS = 'vanillaGradients'
CNFG_SMOOTHGRAD = 'smoothGrad'
CNFG_GUIDED_GRADIENTS = 'guidedGradients'
CNFG_INTEGRATED_GRADIENTS = 'integratedGradients'
CNFG_INPUTS_GRADIENTS = 'inputsGradients'
CNFG_OCCLUSION_SENSITIVITY = 'occlusionSensitivity'

CNFG_INPUT_PATH = 'inputPath'
CNFG_OUTPUT_PATH = 'outputPath'
CNFG_BATCH_SIZE = 'batchSize'
CNFG_NUM_DIM = 'numDim'
CNFG_FOLDS_PATH = 'foldsPath'
CNFG_PATIENT_IDS_PATH = 'patientIdsPath'
CNFG_PATH = 'path'
CNFG_LAYER_NAME = 'layerName'
CNFG_GUIDED_GRADS = 'guidedGrads'
CNFG_INPUT_INDEX = 'inputIndex'
CNFG_NORM_GRADS = 'normGrads'
CNFG_FILTER_VALUES = 'filterValues'
CNFG_STEPS = 'steps'
CNFG_STANDARDIZE = 'standardize'
CNFG_NORMALIZE = 'normalize'
CNFG_THRESHOLD = 'threshold'
CNFG_CONVERT = 'convert'
CNFG_PAD = 'pad'
CNFG_TO_GRAYSCALE = 'to_grayscale'
CNFG_NUM_SAMPLES = 'numSamples'
CNFG_NOISE = 'noise'
CNFG_OCCLUDING_VORTEX = 'occludingVortex'
CNFG_PATCH_SIZE = 'patchSize'
CNFG_TAG = 'tag'

#------------------------------------------------------------------------------

# analyzer wrapper parameters
WRAPPER_BATCH_SIZE = 'batch_size'
WRAPPER_OUTPUT_PATH = 'output_path'
WRAPPER_NDIM = 'ndim'
WRAPPER_HM_STANDARDIZE = 'hm_standardize'
WRAPPER_HM_NORMALIZE = 'hm_normalize'
WRAPPER_HM_THRESHOLD = 'hm_threshold'
WRAPPER_HM_CONVERT = 'hm_convert'
WRAPPER_HM_PAD = 'hm_pad'
WRAPPER_HM_TO_GRAYSCALE = 'hm_to_grayscale'

# losses
LOSS_MEAN = 'mean'
LOSS_MAX = 'max'
LOSS_IDENTITY = 'identity'

# smoothgrad parameters
SMOOTHGRAD_NUM_SAMPLES = 'num_samples'
SMOOTHGRAD_NOISE = 'noise'

# integrated gradients parameters
INTGRADS_STEPS = 'steps'

# occlusion sensitivity parameters
OCCLSENS_OCCLUDING_VORTEX = 'occluding_vortex'
OCCLSENS_PATCH_SIZE = 'patch_size'

# gradcam parameters
GRADCAM_INIDX = 'input_index'
GRADCAM_LAYER_NAME = 'layer_name'
GRADCAM_GUIDED_GRADS = 'guided_grads'
GRADCAM_NORM_GRADS = 'norm_grads'
GRADCAM_FILTER_VALUES = 'filter_values'

# analyzer's tag
ATAG = 'tag'

#------------------------------------------------------------------------------

XDIM = 2 # extra dims: batch and color channel
