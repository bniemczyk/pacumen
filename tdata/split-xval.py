#!/usr/bin/env python

import os
import numpy

# collect the filenames
fnames = []
[fnames.extend([f for f in files]) for root, dirs, files in os.walk('tmp/')]
fnames = numpy.array(fnames)

# randomize the filename order
numpy.random.shuffle(fnames)

xvalcount = fnames.shape[0] / 3

[os.rename('tmp/%s' % (f,), 'xval/%s' % (f,)) for f in fnames[:xvalcount]]
[os.rename('tmp/%s' % (f,), 'train/%s' % (f,)) for f in fnames[xvalcount:]]
