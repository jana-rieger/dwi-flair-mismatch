'''Main module to run visualizations.'''

import numpy as np
import matplotlib.pyplot as plt

from vis.core import utils
from vis.core.vol_viewer import IndexTracker

def main():
    '''Main.'''

    # path to the image to be analyzed
    impath = './data/images/1kplus0003_dwi.nii.gz'
    #impath = './data/images/1kplus0003_flair.nii.gz'

    # path to the directory with heatmaps
    hmpath = './data/heatmaps'
    #hmpath = './data/heatmaps/new_model'
    #hmpath = './data/heatmaps/new_model_split'
    #hmpath = './data/heatmaps/new_model_split_gradcams_norm'
    #hmpath = './data/heatmaps/new_model_split_gradcams_conv'
    #hmpath = './data/heatmaps/new_model_split_gradcampp_conv'


    image, heatmaps, titles = utils.load_data(impath, hmpath)

    # preprocess
    #image = (image * 255).astype("uint8")
    #heatmaps = [np.uint8(heatmap * 255) for heatmap in heatmaps]
    #heatmaps = [utils.interpolate(utils.turbo_colormap_data, heatmap) for heatmap in heatmaps]

    # transpose data
    image = image.T
    heatmaps = [heatmap.T for heatmap in heatmaps]

    # define plot parameters
    cmap = 'turbo'
    imweight = 1
    nrows = 2
    ncols = 5
    # subplot_args = {'nrows': 2, 'ncols': 5, 'figsize': (8, 16),
    #                 'subplot_kw': {'xticks': [], 'yticks': []}}

    utils.remove_keymap_conflicts({'j', 'k'}, plt)
    fig, _ = plt.subplots()
    #fig, _ = plt.subplots(**subplot_args)
    axes = fig.axes

    axes = []
    axes.append(plt.subplot2grid((nrows, ncols), (0, 0)))
    for i in range(nrows):
        for j in range(1, ncols):
            axes.append(plt.subplot2grid((nrows, ncols), (i, j)))

    # ax1 = plt.subplot2grid((3, 3), (0, 0))
    # ax2 = plt.subplot2grid((3, 3), (1, 0))
    # ax3 = plt.subplot2grid((3, 3), (1, 2))
    # ax4 = plt.subplot2grid((3, 3), (2, 0))
    # ax5 = plt.subplot2grid((3, 3), (2, 1))

    tracker = IndexTracker(axes, image, heatmaps, titles, imweight=imweight, cmap=cmap)
    fig.canvas.mpl_connect('key_press_event', tracker.key_press)
    #fig.tight_layout(pad=0.0)
    plt.subplots_adjust(wspace=0.3)
    plt.show()

if __name__ == "__main__":
    main()
