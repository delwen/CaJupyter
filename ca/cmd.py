#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
from __future__ import division

import argparse
import itertools
import numpy as np

class RangedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        origin = values
        arg = map(lambda x: x.strip(), values.split(','))

        def expand_range(maybe_ranged):
            if not ':' in maybe_ranged:
                return [float(maybe_ranged)]

            parts = maybe_ranged.split(':')
            return np.arange(*map(lambda x: float(x.strip()), parts)).tolist()

        values = list(itertools.chain.from_iterable(map(expand_range, arg)))
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + "_str", origin)