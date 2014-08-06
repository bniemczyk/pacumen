# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import scipy
import numpy
import matplotlib.pyplot as plt

# <codecell>

gaussian = numpy.random.normal(size=(100000,2))

# <codecell>

plt.hist2d(gaussian[:,0], gaussian[:,1], 40)
None

# <codecell>

mixer = numpy.random.random((2,2)) * 2 - 1
mixer

# <codecell>

mixed_gaussian = numpy.dot(mixer, gaussian.T).T

# <codecell>

plt.hist2d(mixed_gaussian[:,0], mixed_gaussian[:,1], 40)
None

# <markdowncell>

# $
# L = e^{-\frac{1}{2} \ln(|\Sigma|) - \frac{1}{2}(\mathbf x - \mathbf \mu)^T \Sigma^{-1}(\mathbf x - \mathbf \mu) - \frac{k}{2} \ln(2 \pi)}
# $

# <codecell>

likelihood(x):
    right_term = numpy.log(2 * numpy.pi)
    icov = 

