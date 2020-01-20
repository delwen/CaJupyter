# coding=utf-8
from __future__ import print_function

import numpy as np
import sys

def dff(data, over=None, baseline=10, bg=None):
    l = data.shape[0]
    t = data.shape[1]

    f0 = data[:, 0:baseline].mean(axis=1).reshape((l, 1))

    if over is None:
        over = np.array(f0, copy=True)

    over = np.tile(over, (1, t))
    f0 = np.tile(f0, (1, t))

    if bg is not None:
        # red bg data has t == 1
        if bg.shape[1] == 1:
            bg = np.tile(bg, (1, t))
        f_bg_mean = bg.mean(axis=0)
        f_bg_mean = f_bg_mean.reshape((1, t))
        f_bg = np.tile(f_bg_mean, (l, 1))
        over -= f_bg

    return (data - f0) / over


def split_image(data, n=91):
    nimg = data.shape[1] // n
    slices = [np.s_[i*n:(i+1)*n] for i in range(nimg)]

    def split_red(img):
        return img[:, 0:n-1], img[:, n-1].reshape(data.shape[0], 1)

    return list(map(split_red, [data[:, s] for s in slices]))
