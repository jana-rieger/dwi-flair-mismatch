'''Main module to run analyzers.'''

# import sys
# print(sys.path)
# print(__file__)

import numpy as np

from xAI.src.xai.utils import data
from xAI.src.xai.utils import model
from xAI.src.xai.utils.config.parser import Parser
from xAI.src.xai.wrapper.analyzer import Analyzer

# ugly way to resolve import issue
# import sys
# sys.path.append("../../")

#------------------------------------------------------------------------------

def main():
    '''Main.'''

    # params path
    params_path = './xAI/src/xai/config_dwi_tti.yml'
    parser = Parser(params_path)

    # get dwi input paths
    dwi_paths = sorted(data.get_images(parser.imaging_input, '*dwi*', '.gz'))
    # load models
    dwi_model, _ = model.load_models(parser.model_path)

    # filter images
    # patient_ids = data.get_patient_ids_from_csv(parser.patient_ids_path)
    patient_ids = data.get_patient_ids_from_test_fold(parser.model_path, parser.folds_path)
    filtered_dwi_paths = data.filter_img_paths(img_paths=dwi_paths, filter_list=patient_ids)

    # parse analyzer wrapper params
    wrapper_params = parser.parse_analyzer_wrapper()

    # instantiate a dwi analyzer
    print('Analyzing the DWI model...')
    analyzers = parser.parse_analyzers(dwi_model)
    dwi_analyzer = Analyzer(analyzers=analyzers, imaging_data=[filtered_dwi_paths], numeric_data=None, preprocess='dwi',
                            **wrapper_params)
    dwi_analyzer.compute_heatmaps()


if __name__ == "__main__":
    main()
