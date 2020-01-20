from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import defaultdict


def get_nb_condition(df):
    conds = np.unique(df.Condition)
    candidates = [x for x in conds if x.startswith("noise")]
    if len(candidates) > 0:
        return candidates[0]
    return candidates[0] if len(candidates) > 0 else "noisebox"


def load_exclude(path):
    import csv
    excludes = defaultdict(list)

    if path is None:
        return excludes

    with open(path, 'r') as fd:
        ex = csv.reader(fd, delimiter=' ', quotechar='"')
        for row in ex:
            ncols = len(row)
            if ncols < 1:
                continue
            excludes[row[0]] += [{"image": row[1], "pulse": row[2]}]

    #for k, v in excludes.items():
    #    for ex in v:
    #       print('%s %s %s' % (k, ex['image'], ex['pulse']))

    return excludes


def index_of_name(lst, name):
    for idx, entity in enumerate(lst):
        if entity.data.name == name:
            return idx
    return -1


def neuron_name(nid):
    return "%d%c" % (int(nid) >> 8, chr(ord('a') + int(nid) & 0xFF))


def neuron_dendrite_length(full, neuron):
    n = None
    for n in full.data_arrays:
        if n.name.startswith(neuron):
            break
        n = None
    if n is None:
        return np.NaN
    return n.shape[0]


age_to_color = {10: [0xFF, 0xAA, 0x00],
                11: [0xCC, 0x00, 0x52],
                13: [0x00, 0xCC, 0x67],
                14: [0x00, 0x87, 0xCC],
                15: [0x00, 0x00, 0x99],
                16: [0x99, 0x99, 0x99],
                17: [0xCC, 0x88, 0x00],
                18: [0x99, 0x99, 0x7A],
                60: [0xFF, 0x55, 0x00]}


def col_age_to_color(col):
    for a in col['Age']:
        yield list(map(lambda x: x/0xFF, age_to_color[a]))

