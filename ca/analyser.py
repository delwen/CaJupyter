
from collections import defaultdict

import nixio as nix
import time
import numpy as np

from ca.nix import *
from ca.img import *

from ca.experiment import *


class CaAnalyser(object):
    def __init__(self, root, over, baseline, dlen, cut, excludes, pframe):
        self.root = root
        self.excludes = excludes or {}
        self.filelist = None
        self.bsl = baseline
        self.dlen = dlen
        self.cut = cut
        self.over = over
        self.pframe = pframe
        # NIX outfile related things
        self.filename = None
        self.nf = None
        self.dff_full = None
        self.dff_mean = None
        self.dff_peak = None
        self.peaks = None
        self.avgs = None
        self.ages = None
        self.means = None
        self.fcnd = None  # neuron condition [feature]
        self.fage = None  # neuron age [feature]
        self.fneu = None  # neuron name [feature]
        self.dff_full_pulses = {}  # groups for condition
        # Current processing state
        self.neuron = None
        self.image = None
        self.idx = 0
        self.loop = None
        # helper stuff
        self.chmap =list(map(chr, range(ord('a'), ord('z')+1)))
        # for background correction
        self.bg = None
        # for warning/error reporting
        self.warnings = []
        self.messages = []

    def print(self, msg, *args, **kwargs):
        self.messages.append(msg)

    def setup(self):
        self.filelist = find_files_recursive(self.root, "[!c]*.nix")
        if len(self.filelist) == 0:
            return
        peak_frame = 'full'
        pst, pnd = self.pframe
        if pst is not None or pnd is not None:
            peak_frame = "%s_%s" % ((pst or 0), (pnd or "end"))

        outname = "ca-%s-bs%d-%s-%s-%s.nix" % \
                  (self.over,
                   self.bsl,
                   str(self.dlen or 'auto') + str("c" if self.cut else "p"),
                   str(self.bg or 'uc'),
                   peak_frame)
        self.filename = outname
        self.nf = nix.File.open(outname, nix.FileMode.Overwrite)
        self.dff_full = self.nf.create_block("full", "dff.full")
        self.dff_mean = self.nf.create_block("mean", "dff.mean")
        self.dff_peak = self.nf.create_block("peak", "dff.peak")

        b = self.dff_peak
        self.peaks = [b.create_data_array("ap%d.all" % p, "pulse.%d.max" % p,
                                          dtype='d', shape=(0, 2)) for p in pulses]
        for da in self.peaks:
            da.append_set_dimension()
            da.append_set_dimension()

        self.avgs = [b.create_data_array("ap%d.avg" % p, "pulse.%d.max.avg" % p,
                                         dtype='d', shape=(0,)) for p in pulses]
        for da in self.avgs:
            da.append_set_dimension()

        self.ages = {}

        for p in pulses:
            for a in ages:
                name = "P%d.ap%d" % (a, p)
                da = b.create_data_array(name, "nix.position", dtype='d', shape=(0,))
                da.append_set_dimension()
                self.ages[name] = da

                mtag = b.create_multi_tag(name, "pulse.avg@age", da)
                mtag.references.append(self.avgs[pulses.index(p)])

        self.fage = b.create_data_array("age", "feature.age", dtype='i', shape=(0,))
        self.fage.append_set_dimension()

        self.fcnd = b.create_data_array("condition", "feature.condition", dtype='i', shape=(0,))
        self.fcnd.append_set_dimension()

        self.fneu = b.create_data_array("neuron", "feature.neuron", dtype='u8', shape=(0,))
        self.fneu.append_set_dimension()

        self.means = b.create_data_array("peak.averages", "nix.position", dtype='d', shape=(0,))

        mtag = b.create_multi_tag("peak.averages", "pulse.avg", self.means)
        for i, da in enumerate(self.avgs):
            mtag.references.append(da)

        mtag.create_feature(self.fage, nix.LinkType.Tagged)
        mtag.create_feature(self.fcnd, nix.LinkType.Tagged)
        mtag.create_feature(self.fneu, nix.LinkType.Tagged)

        s = self.nf.create_section('params', 'params')
        s['baseline'] = self.bsl
        s['cut'] = 1 if self.cut else 0
        s['length'] = self.dlen or 'auto'
        s['over'] = self.over
        if self.bg is not None:
            s['bg-correction'] = self.bg

        if pst is not None:
            s['peak-start'] = pst
        if pnd is not None:
            s['peak-end'] = pnd

        b = self.dff_full

        #for p in pulses:
        #   self.dff_full_pulses[p] = b.create_group("ap%d" % p, "pulse.%d" % p)

    def should_exclude_subimage(self, neuron, image, subimage):
        if neuron not in self.excludes:
            return False
        ex = self.excludes[neuron]
        for i in ex:
            if i['image'] == str(image) and i['pulse'] == str(subimage):
                return True
        return False

    def process(self):
        for path in self.filelist:
            self.process_file(path)

    def process_file(self, path):
        try:
            cf = nix.File.open(path, nix.FileMode.ReadOnly)
        except:
            print("[W] could not open %s" % path)
            return
        neurons = [b for b in cf.blocks if b.type == 'neuron']

        for neuron in neurons:
            name = neuron.name
            self.neuron = neuron

            exclude = self.should_exclude_subimage(name, '*', '*')
            msg = name
            if exclude:
                msg += " skipping (exclude file)"

            msg = (u'├── ' if exclude else u'├─┬ ') + msg
            self.print(msg.encode('utf-8'))
            if exclude:
                time.sleep(0.5)
                self.neuron = None
                continue

            self.process_neuron(neuron)

        cf.close()
        cf = None

    def neuron_id(self, name):
        date = int(name[:8])
        suffix = self.chmap.index(name[8:])
        return date << 8 | suffix

    def process_neuron(self, neuron):
        name = neuron.name
        age = int(neuron.metadata['age'])
        cnd = int(neuron.metadata['condition'].lower() == 'noisebox')

        self.neuron_meta = self.nf.create_section(name, 'ca.neuron')
        self.neuron_meta['age'] = neuron.metadata['age']
        self.neuron_meta['condition'] = neuron.metadata['condition']

        images = sorted(items_of_type(neuron.groups, "image.ca"),
                        key=lambda x: x.metadata['creation_time'])

        pos_loop = defaultdict(list)

        for image in images:
            if self.should_exclude_subimage(name, image.name, '*'):
                msg = u'│ ├── image: %s excluded. Skipping!' % image.name
                self.print(msg.encode('utf-8'))
                continue

            pos = self.process_image(image)
            for l, p in pos.items():
                pos_loop[l] += p

        self.print(u'│ ├─ Tags: '.encode('utf-8'), end='')
        dff_peak = self.dff_peak

        mean_ap = [np.nan for _ in pulses]
        for l, p in pos_loop.items():
            self.print("ap%d " % l, end='')
            pdat = np.zeros((len(p), 2))
            pdat[:, 0] = p
            aps = "ap%d" % l
            da_name = "%s.%s" % (self.neuron.name, aps)
            pos = dff_peak.create_data_array(da_name + ".all", "nix.positions", data=pdat)
            pos.append_set_dimension()
            pos.append_set_dimension()

            mtag = self.dff_peak.create_multi_tag(da_name, aps, pos)
            mtag.references.append(self.peaks[pulses.index(l)])

            # calculate the mean per neuron, per pulse
            pidx = pulses.index(l)
            data = np.array([mtag.retrieve_data(pc, 0)[0] for pc in range(len(p))])
            avg_data = data.mean()
            mean_ap[pidx] = avg_data
            # TODO: dimensions, metadata

        self.print(u'\n│ ├─ Means: '.encode('utf-8'), end='')
        for i, m in enumerate(mean_ap):
            avg = self.avgs[i]
            avg_pos = [avg.shape[0]]
            avg.append(m)
            key = "P%d.ap%d" % (age, pulses[i])
            self.ages[key].append(np.array(avg_pos))  # .reshape((1, )), axis=0)
            self.print("ap%d: %3.2f @ %d; " % (pulses[i], m, avg_pos[0]), end='')

        self.means.append(np.array(self.means.shape[0]))
        self.fage.append(np.array([age]))
        self.fcnd.append(np.array([cnd]))
        self.fneu.append(np.array([self.neuron_id(name)]))
        self.print('')

    def maybe_background(self, das):
        if self.bg is None:
            return None

        bgs = items_of_type(das, 'kymo.bg')
        target = '%s.kymo.%s' % (self.image.name, self.bg)
        matches = [bg for bg in bgs if bg.name == target]
        if len(matches) != 1:
            msg = " %s/%s: could not find exact background '%s' (%d)" \
                  % (self.neuron.name, self.image.name, self.bg, len(matches))
            print("[WARNING] %s\n" % msg)
            self.warnings += [msg]
            return None
        return matches[0]

    def process_image(self, image):
        self.image = image
        meta = image.metadata
        das = image.data_arrays
        channels = item_of_type(das, "channel")
        red = list(np.nonzero(np.array(channels) == 1)[0])
        kymo = item_of_type(das, 'kymo.fg')
        bgko = self.maybe_background(das)
        condition = meta['condition']
        loop = loops.get(condition, None)
        if loop is None:
            msg = u'│ ├──  WARN: unknown loop: %s. skipping!' % condition
            self.print(msg.encode('utf-8'))
            return
        else:
            msg = u'│ ├── image: %s, Loop: %s' % (image.name, condition)
            self.print(msg.encode('utf-8'), end='')
        if len(loop) != len(red):
            self.print(' [len(loops) != len(condition)]', end='')

        imgs = split_image(kymo[:])
        bgs = split_image(bgko[:]) if bgko is not None else [(None, None)]*len(imgs)
        self.loop = loop

        pos_loop = defaultdict(list)

        self.print(" [", end='')
        for idx, l in enumerate(loop[:len(red)]):
            self.print('%2d: ' % self.loop[idx], end='')
            if self.should_exclude_subimage(self.neuron.name, image.name, idx+1):
                self.print('SK; ', end='')
                continue

            img = imgs[idx]
            bg = bgs[idx]

            bg_data = bg[1] if self.over == 'red' else bg[0]
            over_data = img[1] if self.over == 'red' else None
            dff_data = dff(img[0], over_data, baseline=self.bsl, bg=bg_data)

            if self.dlen is not None:
                if dff_data.shape[0] < np.abs(self.dlen):
                    self.print('SD; ', end='')
                    continue
                if self.cut:
                    dff_data = dff_data[:self.dlen, :]

            di = self.save_dff_full(idx, data=dff_data)

            dff_mean = np.array(dff_data).mean(axis=0)
            da_mean = self.save_dff_mean(idx, data=dff_mean)
            if self.pframe[1] is not None:
                pend = self.pframe[1]
                dff_mean = dff_mean[:pend]
            if self.pframe[0] is not None:
                pstart = self.pframe[0]
                dff_mean = dff_mean[pstart:]

            peak_idx = np.argmax(dff_mean)
            peak_val = dff_mean[peak_idx]

            da_peaks = self.peaks[pulses.index(l)]

            pos_loop[l] += [da_peaks.shape[0]]
            da_peaks.append(np.array([peak_val, peak_idx]).reshape((1, 2)), axis=0)

            # group handling
            # self.dff_full_pulses[l].data_arrays.append(di)

            # metadata handling
            if False:
                mi = self.neuron_meta.create_section(di.name, di.type)
                mi['condition'] = l
                di.metadata = mi
            self.print('ok; ', end='')

        self.print('] ')
        return pos_loop

    def finish(self):
        if self.nf is not None:
            self.dff_full = None
            self.dff_mean = None
            self.nf.close()

    def save_dff_full(self, idx, data):
        suffix = self.over
        da_name = "%s.ap%0.2d.%s.%d.%s" % (self.neuron.name, self.loop[idx], self.image.name, idx+1, suffix)
        da_type = "dff." + suffix
        da = self.dff_full.create_data_array(da_name, da_type, data=data)
        da.label = 'dF/F (%s)' % suffix

        dim = da.append_sampled_dimension(1)
        dim.label = 'location'

        dim = da.append_sampled_dimension(30)
        dim.label = 'time'
        dim.unit = 'ms'
        # TODO: offset

        return da

    def save_dff_mean(self, idx, data):
        suffix = self.over
        da_name = "%s.ap%0.2d.%s.%d.%s" % (self.neuron.name, self.loop[idx], self.image.name, idx+1, suffix)
        da_type = "dff.mean." + suffix
        da = self.dff_mean.create_data_array(da_name, da_type, data=data)
        da.label = 'mean dF/F (%s)' % suffix

        dim = da.append_sampled_dimension(30)
        dim.label = 'time'
        dim.unit = 'ms'
        # TODO: offset

        return da