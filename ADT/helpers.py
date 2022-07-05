import math
import os
import re
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

# os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def downsample_with_averaging(array):
    """
    Downsample x by factor using averaging.
    @return: The downsampled array, of the same type as x.
    """
    # TODO: needs to be adapted to the shape of the input array
    if len(array.shape) == 3:
        factor = (2, 2, 1)
    else:
        factor = (2, 2)

    if np.array_equal(factor[:3], np.array([1, 1, 1])):
        return array

    output_shape = tuple(int(math.ceil(s / f)) for s, f in zip(array.shape, factor))
    temp = np.zeros(output_shape, float)
    counts = np.zeros(output_shape, np.int)
    for offset in np.ndindex(factor):
        part = array[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
        indexing_expr = tuple(np.s_[:s] for s in part.shape)
        temp[indexing_expr] += part
        counts[indexing_expr] += 1
    return np.cast[array.dtype](temp / counts)


def downsample_with_max_pooling(array):
    # TODO: needs to be adapted to the shape of the input array
    factor = (2, 2)

    if np.all(np.array(factor, int) == 1):
        return array

    sections = []

    for offset in np.ndindex(factor):
        part = array[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
        sections.append(part)

    output = sections[0].copy()

    for section in sections[1:]:
        np.maximum(output, section, output)

    return output


def striding(array):
    """Downsample x by factor using striding.
    @return: The downsampled array, of the same type as x.
    """
    # TODO: needs to be adapted to the shape of the input array
    factor = (2, 2)
    if np.all(np.array(factor, int) == 1):
        return array
    return array[tuple(np.s_[::f] for f in factor)]


def make_folder(path):
    if not os.path.exists(path):
        print('Creating folder:', path)
        os.makedirs(path)
    return path


def plot_loss_acc_history(history, epochs, suptitle='', save_name='', val=True, save=True, draw_line=10):
    """
    Plots loss and performance meassure curves from training.
    :param results: Lists of history data for each epoch from the model training.
    :param suptitle: String, main title of the plot.
    :param save_name: String, under what name to save the plot.
    :param val: True if the data from validation shall be plotted as well. False otherwise.
    :param save: True of the plot shall be saved as png image.
    :return:
    """
    print('Plotting loss and accuracy ...')
    plt.figure(figsize=(10, 5))
    plt.suptitle("\n".join(wrap(suptitle, 80)))
    # to make the values on x axis start from 1 and not 0
    x_dim = np.arange(epochs, dtype=int) + 1

    # subplot for plotting the loss values during training
    plt.subplot(1, 2, 1)
    plt.title('train loss')
    plt.axvline(x=draw_line, color='red', linestyle='dotted')
    plt.plot(x_dim, history['loss'], color='blue', label='train loss')

    if val:
        plt.plot(x_dim, history['val_loss'], color='orange', label='val loss')
        plt.title('train vs validation loss')

    all_values = history['loss'] + history['val_loss']
    median = np.median(all_values)
    maxv = np.max(all_values)
    if median / maxv < 0.2:
        plt.ylim(bottom=np.min(all_values) * 0.9, top=median * 2.5)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)

    # subplot for plotting the performance measure values during the training
    if 'auc' in history:
        metric = 'auc'
    elif 'accuracy' in history:
        metric = 'accuracy'
    elif 'acc' in history:
        metric = 'acc'
    elif 'mae' in history:
        metric = 'mae'
    else:
        metric = None
    if metric is not None:
        plt.subplot(1, 2, 2)
        plt.title('train ' + metric)
        plt.axvline(x=draw_line, color='red', linestyle='dotted')
        plt.plot(x_dim, history[metric], color='blue', label='train ' + metric)

        if val:
            plt.plot(x_dim, history['val_' + metric], color='orange', label='val ' + metric)
            plt.title('train vs validation ' + metric)

        all_values = history[metric] + history['val_' + metric]
        median = np.median(all_values)
        maxv = np.max(all_values)
        if median / maxv < 0.2:
            plt.ylim(bottom=np.min(all_values) * 0.9, top=median * 2.5)

        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    if save:
        plt.savefig(save_name)

    plt.close()


# noinspection PyTypeChecker
def multi_slice_viewer(volumes, titles, cmap=None, suptitle='', norm=None, start=50, nr_rotations=-1, axis=2):
    # used code from tutorial: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
    def remove_keymap_conflicts(new_keys_set):
        """Remove chosen keys from matplotlib’s default key maps to prevent conflicts."""
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def process_key(event):
        """Bind events to key press."""
        figure = event.canvas.figure
        for ax in figure.axes:
            if event.key == 'j':
                previous_slice(figure, ax)
            elif event.key == 'k':
                next_slice(figure, ax)
            figure.canvas.draw()

    def previous_slice(figure, ax):
        """Go to the previous slice."""
        volume = ax.volume
        if ax.index > 0:
            ax.index = (ax.index - 1)
            ax.images[0].set_array(np.take(volume, ax.index, axis=axis))
            for t in range(len(figure.texts)):
                if t > 1:
                    figure.texts[t].set_visible(False)
            figure.text(0.95, 0.84, 'Slice ' + str(ax.index), ha='right', va='center', size=10, color='red')

    def next_slice(figure, ax):
        """Go to the next slice."""
        volume = ax.volume
        if ax.index < volume.shape[0] - 1:
            ax.index = (ax.index + 1)
            ax.images[0].set_array(np.take(volume, ax.index, axis=axis))
            for t in range(len(figure.texts)):
                if t > 1:
                    figure.texts[t].set_visible(False)
            figure.text(0.95, 0.84, 'Slice ' + str(ax.index), ha='right', va='center', size=10, color='red')

    print('Plotting...')
    if cmap is None:
        cmap = []
        for i in range(len(volumes)):
            cmap.append('gray')
    if norm is None:
        norm = []
        for i in range(len(volumes)):
            norm.append(None)

    remove_keymap_conflicts({'j', 'k'})
    figure, axes = plt.subplots(1, len(volumes), figsize=(10, 5), sharey=True)
    figure.suptitle(suptitle)
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.75, left=0.05, right=0.95)
    for j in range(len(axes)):
        volume = np.rot90(volumes[j], nr_rotations)
        axes[j].volume = volume
        axes[j].index = start
        axes[j].imshow(np.take(volume, axes[j].index, axis=axis), cmap=cmap[j], norm=norm[j])
        axes[j].set_title(titles[j])
        # axes[j].set_axis_off()
    figure.text(0.05, 0.84, 'Press \'j\' for previous slice, \'k\' for next slice', ha='left', va='center', size=10,
                color='red')
    figure.text(0.95, 0.84, 'Slice ' + str(start), ha='right', va='center', size=10, color='red')
    figure.canvas.mpl_connect('key_press_event', process_key)


def unique_and_count(array, name='', print_dict=True):
    unique, count = np.unique(array, return_counts=True)
    print(name, 'unique:', unique, 'size:', len(unique))
    if print_dict:
        print(name, dict(zip(unique, count)))
    return unique, count


class EndswithDict(dict):

    def __init__(self, *args, **kwargs):
        super(EndswithDict, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        for k, v in self.items():
            if item.endswith(k):
                return v
        raise KeyError


class StartswithDict(dict):

    def __init__(self, *args, **kwargs):
        super(StartswithDict, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        for k, v in self.items():
            if item.startswith(k):
                return v
        raise KeyError


if __name__ == '__main__':
    rd1 = EndswithDict({'shift_dicho_release': 3, 'baseline': 1, 'release': 2, })
    print(rd1)

    lst = ['baseline',
           'baseline_image_ext',
           'release',
           'release_img_ext',
           'shift_dicho_baseline',
           'shift_dicho_baseline_image_ext',
           'shift_dicho_release',
           'shift_dicho_release_image_ext',
           'full_range_baseline',
           'full_range_baseline_image_ext']

    for l in lst:
        try:
            print(l, rd1[l])
        except KeyError:
            print(l, 'not found')

    rd2 = StartswithDict({'dicho': 2, 'full': 6})
    print(rd1)

    lst = ['dicho_baseline',
           'dicho_baseline_image_ext',
           'dicho_release',
           'dicho_release_img_ext',
           'dicho_shift_dicho_baseline',
           'dicho_shift_dicho_baseline_image_ext',
           'dicho_shift_dicho_release',
           'dicho_shift_dicho_release_image_ext',
           'full_baseline',
           'full_baseline_image_ext']

    for l in lst:
        try:
            print(l, rd2[l])
        except KeyError:
            print(l, 'not found')
