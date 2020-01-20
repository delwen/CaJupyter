# coding=utf-8

import os
import fnmatch

def items_of_type(lst, the_type):
    return [x for x in lst if x.type == the_type]


def item_of_type(lst, the_type):
    items = items_of_type(lst, the_type)
    if len(items) != 1:
        raise RuntimeError("Bleh %d" % len(items))
    return items[0]

def find_files_recursive(dir, glob):
    entries = os.listdir(dir)
    res = []
    for e in entries:
        fn = os.path.join(dir, e)
        if os.path.isdir(fn):
            res += find_files_recursive(fn, glob)
        elif fnmatch.fnmatch(e, glob):
            res += [fn]
    res = list(map(os.path.abspath, res))
    return res
