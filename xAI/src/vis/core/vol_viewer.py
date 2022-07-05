'''3D Visualizer.'''

from typing import List
import numpy as np

class IndexTracker(object):
    '''3D Volumetric Data Viewer.'''

    def __init__(self,
                 axes: List,
                 image: np.ndarray,
                 heatmaps: List[np.ndarray],
                 titles: List[str],
                 imweight: float = 0.5,
                 cmap: str = 'turbo'):
        '''
        Args:
            axes (list[]): List of matplotlib's axes.
            image (np.ndarray): Image to be visualized, of dim (D, H, W).
            heatmaps (list[np.ndarray]): List of heatmas to be visualized, of dim (D, H, W).
            titles (list[str]): List of titles for each heatmap.
            cmap (str): Colormap used for heatmap visualization.
        '''

        self.axes = axes
        self.image = image
        self.heatmaps = heatmaps
        self.titles = titles
        self.slices = image.shape[0]
        self.ind = self.slices // 2 # middle slice

        self.axes[0].set_title('image')
        self.axes[0].imshow(self.image[self.ind], cmap='gray')

        for i, (title, heatmap) in enumerate(zip(self.titles, self.heatmaps)):
            self.axes[i+1].set_title(title)
            self.axes[i+1].imshow(self.image[self.ind], cmap='gray')
            self.axes[i+1].imshow(heatmap[self.ind], cmap=cmap, alpha=imweight)

        self.update()

    def key_press(self, event):
        '''
        Updates slice's index.
        '''

        fig = event.canvas.figure

        if event.key == 'j':
            self.ind = (self.ind - 1) % self.slices
        elif event.key == 'k':
            self.ind = (self.ind + 1) % self.slices

        self.update()
        fig.canvas.draw_idle()

    def update(self):
        '''Visualizes current slices'''

        self.axes[0].images[0].set_data(self.image[self.ind])

        for i, heatmap in enumerate(self.heatmaps):
            self.axes[i+1].images[0].set_data(self.image[self.ind])
            self.axes[i+1].images[1].set_data(heatmap[self.ind])
            self.axes[i+1].set_ylabel(f'slice {self.ind}')
            #axes[i].images.axes.figure.canvas.draw()
