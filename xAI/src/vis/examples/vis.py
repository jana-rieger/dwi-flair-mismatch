import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import cm


class IndexTracker(object):
    '''3D Volumetric Data Viewer.'''

    def __init__(self, axes: list, image: np.ndarray, heatmaps: list, titles: list):
        '''
        Args:
            axes (list<>):
            image (np.ndarray):
            heatmaps (list<>):
            titles (list<>):
        '''

        self.axes = axes
        self.image = image
        self.heatmaps = heatmaps
        self.titles = titles
        self.slices = image.shape[0]
        self.ind = self.slices // 2 # middle slice

        for i, (title, heatmap) in enumerate(zip(self.titles, self.heatmaps)):
            self.axes[i].set_title(title)
            self.axes[i].imshow(self.image[self.ind], cmap="gray")
            self.axes[i].imshow(heatmap[self.ind], cmap='jet', alpha=0.5)
        
        self.update()

    def key_press(self, event):
        '''
        TODO
        '''

        fig = event.canvas.figure

        if event.key == 'j':
            self.ind = (self.ind - 1) % self.slices
        elif event.key == 'k':
            self.ind = (self.ind + 1) % self.slices

        self.update()
        fig.canvas.draw_idle()

    def update(self):
        '''
        TODO
        '''

        for i, heatmap in enumerate(self.heatmaps):
            self.axes[i].images[0].set_data(self.image[self.ind])
            self.axes[i].images[1].set_data(heatmap[self.ind])
            self.axes[i].set_ylabel(f'slice {self.ind}')
            #axes[i].images.axes.figure.canvas.draw()


def remove_keymap_conflicts(new_keys_set):
    '''
    '''

    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def get_titles(hmpaths):
    '''
    TODO
    '''

    titles = [os.path.basename(hmpath).split('_')[0] for hmpath in hmpaths]

    return titles


# -------------------------------------------------------------------------------

def main():
    '''Main.'''

    # path to the image to be analyzed
    impath = './data/images/1kplus0003_dwi.nii.gz'
    # path to the directory with heatmaps
    hmpath = './data/heatmaps'

    # load the mage
    image = nib.load(impath).get_fdata()
    # convert to uint 255 and normalize
    #image = (image * 255).astype("uint8")

    # get heatmap paths for the image being investigated
    # each heatmap will match with the image by '*[image_name]'
    imname = os.path.basename(impath)
    hmpaths = glob.glob(os.path.join(hmpath, '*' + imname))
    # load heatmaps
    heatmaps = [nib.load(hmpath).get_fdata() for hmpath in hmpaths]
    heatmaps = [np.uint8(heatmap * 255) for heatmap in heatmaps]

    # transpose data
    image = image.T
    heatmaps = [heatmap.T for heatmap in heatmaps]

    # define plot parameters
    color_map = 'jet'
    image_weight = 0.5
    subplot_args = { 'nrows': 2, 'ncols': 4, 'figsize': (8, 16),
                    'subplot_kw': {'xticks': [], 'yticks': []} }

    #titles = ['GradCAM']
    titles = get_titles(hmpaths)

    remove_keymap_conflicts({'j', 'k'})
    #fig, _ = plt.subplots(1, 1)
    fig, _ = plt.subplots(**subplot_args)
    axes = fig.axes

    tracker = IndexTracker(axes, image, heatmaps, titles)

    fig.canvas.mpl_connect('key_press_event', tracker.key_press)
    plt.show()

if __name__ == "__main__":
    main()



# def multi_slice_viewer(image, heatmap):
#     remove_keymap_conflicts({'j', 'k'})
    
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    
#     print('num axes:', len(fig.axes))
#     print('ax:', ax)
#     print('images:', len(ax.images))
    
#     ax.image = image
#     ax.index = image.shape[0] // 2
#     ax.imshow(image[ax.index], cmap="gray")
    
#     #ax.set_title(titles[0], fontsize=10)
#     heatmap = np.uint8(cm.jet(heatmap) * 255)
#     ax.heatmap = heatmap
#     #ax.index = heatmap.shape[0] // 2
#     ax.imshow(heatmap[ax.index], cmap='jet', alpha=0.5)
    
#     print('images:', len(ax.images))
    
#     #fig, ax = plt.subplots(**subplot_args)
#     #for i, title in enumerate(titles):
#         #heatmap = np.uint8(cm.jet(heatmap) * 255)
    
#     #   ax[i].volume = volume
#     #   ax[i].index = volume.shape[0] // 2
#     #   ax[i].imshow(volume[ax[i].index], cmap="gray")
        
#         #ax[i].set_title(title, fontsize=14)
#         #ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
#     #plt.tight_layout()
#     fig.canvas.mpl_connect('key_press_event', process_key)

# def process_key(event):
#     fig = event.canvas.figure
#     ax = fig.axes[0]

#     if event.key == 'j':
#         previous_slice(ax)
#     elif event.key == 'k':
#         next_slice(ax)
#     fig.canvas.draw_idle()

# def previous_slice(ax):
#     volume = ax.image
#     ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
#     ax.images[0].set_array(volume[ax.index])
    
#     volume = ax.heatmap
#     #ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
#     ax.images[1].set_array(volume[ax.index])

# def next_slice(ax):
#     volume = ax.image
#     ax.index = (ax.index + 1) % volume.shape[0]
#     ax.images[0].set_array(volume[ax.index])
    
#     volume = ax.heatmap
#     #ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
#     ax.images[1].set_array(volume[ax.index])
   

# # move depth dims to the first osition
# image = np.moveaxis(image, -1, 0)
# heatmaps = [np.moveaxis(heatmap, -1, 0) for heatmap in heatmaps]