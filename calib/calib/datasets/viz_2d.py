"""
2D visualization primitives based on Matplotlib.

1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from copy import deepcopy
import math
import cv2
from calib.calib.datasets.view import read_image, resize_image


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True, autoscale=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
        if not autoscale:
            ax[i].autoscale(False)
    fig.tight_layout(pad=pad)
    return plt


def plot_keypoints(kpts, colors='lime', ps=6):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        if k is not None:
            a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def add_text(idx, text, pos=(0.01, 0.99), fs=15, color='w',
             lcolor='k', lwidth=2):
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, va='top', ha='left',
                color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects([
            path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
            path_effects.Normal()])


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches='tight', pad_inches=0, **kw)


def features_to_RGB(*Fs, skip=1):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    flatten = []
    shapes = []
    for F in Fs:
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F = F.reshape(-1, c)
        flatten.append(F)
        shapes.append((h, w))
    flatten = np.concatenate(flatten, axis=0)

    pca = PCA(n_components=3)
    if skip > 1:
        pca.fit(normalize(flatten[::skip]))
        flatten = normalize(pca.transform(normalize(flatten)))
    else:
        flatten = normalize(pca.fit_transform(normalize(flatten)))
    flatten = (flatten + 1) / 2

    Fs = []
    for h, w in shapes:
        F, flatten = np.split(flatten, [h*w], axis=0)
        F = F.reshape((h, w, 3))
        Fs.append(F)
    assert flatten.shape[0] == 0
    return Fs


def plot_row(dict_list_main, pred_annotate=['roll', 'rho', 'fov'], titles=[]):
    dict_list = deepcopy(dict_list_main)

    ims = []
    texts1 = []
    texts2 = []
    texts3 = []
    texts4 = []
    texts5 = []
    texts6 = []
    texts7 = []
    texts8 = []
    for j in range(len(dict_list)):
        im = np.ascontiguousarray(dict_list[j]['image'] * 255, dtype=np.uint8)
        original_im_shape = cv2.imread(dict_list[j]['path'][0]).shape
        im = resize_image(im, width=int(
            round(original_im_shape[1]/4)), height=int(round(original_im_shape[0]/4)))
        ims.append(im)
        if 'roll' in pred_annotate or 'rho' in pred_annotate:
            pred_angle = dict_list[j]['pred_roll'] * \
                np.pi/180 if 'roll' in pred_annotate else 0
            distorted_offset = dict_list[j]['pred_rho'] * \
                im.shape[0] if 'rho' in pred_annotate else 0
            radius = 5000
            pred_centrex = int(round(im.shape[1]/2))
            pred_centrey = int(round(im.shape[0]/2)) - distorted_offset
            pred_x1 = math.floor(math.cos(pred_angle) * radius + pred_centrex)
            pred_y1 = math.floor(math.sin(pred_angle) * radius + pred_centrey)
            pred_x2 = math.ceil(math.cos(pred_angle+np.pi)
                                * radius + pred_centrex)
            pred_y2 = math.ceil(math.sin(pred_angle+np.pi)
                                * radius + pred_centrey)
            cv2.line(im, (pred_x2, pred_y2),
                     (pred_x1, pred_y1), (255, 0, 0), 2)

        if 'roll' in pred_annotate:
            texts1.append({'idx': j, 'text': f"roll (pred): {pred_angle*180/np.pi:.3f}°",
                          'pos': (0.01, 0.99), 'fs': 15, 'color': '#00ff00', 'lcolor': 'k', 'lwidth': 2})
        if 'rho' in pred_annotate:
            texts2.append({'idx': j, 'text': f"rho (pred): {dict_list[j]['pred_rho']:.3f} ratio",
                           'pos': (0.01, 0.92), 'fs': 15, 'color': '#00ff00', 'lcolor': 'k', 'lwidth': 2})
        if 'fov' in pred_annotate:
            texts3.append({'idx': j, 'text': f"fov (pred): {dict_list[j]['pred_fov']:.3f}°",
                           'pos': (0.01, 0.85), 'fs': 15, 'color': '#00ff00', 'lcolor': 'k', 'lwidth': 2})
        if 'k1_hat' in pred_annotate:
            texts4.append({'idx': j, 'text': f"k1_hat (pred): {(dict_list[j]['pred_k1_hat']):.3f}",
                           'pos': (0.01, 0.78), 'fs': 15, 'color': '#00ff00', 'lcolor': 'k', 'lwidth': 2})

    plot_images(ims, titles=titles)
    for j in range(len(ims)):
        if 'roll' in pred_annotate:
            add_text(**texts1[j])
        if 'rho' in pred_annotate:
            add_text(**texts2[j])
        if 'fov' in pred_annotate:
            add_text(**texts3[j])
        if 'k1_hat' in pred_annotate:
            add_text(**texts4[j])
    return plt
