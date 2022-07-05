'''Main module to run visualizations.'''

import numpy as np
import matplotlib.pyplot as plt

import utils
from vol_viewer import IndexTracker

def main():
    '''Main.'''

    # path to the image to be analyzed
    impath = './data/images/1kplus0003_flair.nii.gz'
    # path to the directory with heatmaps
    hmpath = './data/heatmaps'

    image, heatmaps, titles = utils.load_data(impath, hmpath)

    # preprocess
    #image = (image * 255).astype("uint8")
    #heatmaps = [np.uint8(heatmap * 255) for heatmap in heatmaps]

    # transpose data
    image = image.T
    heatmaps = [heatmap.T for heatmap in heatmaps]

    # define plot parameters
    color_map = 'jet'
    image_weight = 0.5
    subplot_args = {'nrows': 2, 'ncols': 4, 'figsize': (8, 16),
                    'subplot_kw': {'xticks': [], 'yticks': []}}

    utils.remove_keymap_conflicts({'j', 'k'}, plt)
    #fig, _ = plt.subplots(1, 1)
    fig, _ = plt.subplots(**subplot_args)
    axes = fig.axes

    tracker = IndexTracker(axes, image, heatmaps, titles)
    fig.canvas.mpl_connect('key_press_event', tracker.key_press)
    plt.show()

if __name__ == "__main__":
    main()
