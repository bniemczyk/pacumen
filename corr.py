#!/usr/bin/env python

import oneclasstree
import pacumen
import numpy
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import sys
import cPickle as pickle

classfiles = sys.argv[1:]
classifiers = {x.split('/')[-1]: pickle.load(open(x)) for x in classfiles}
print classifiers.keys()

for c in classifiers:
  columns = classifiers[c].selected_columns()
  fvecs = pacumen.get_class_fvecs(c, remove_zero=False)
  fvecs = fvecs.tolil()[:,tuple(columns)].todense()
  fvecs = fvecs - fvecs.mean(axis=0)
  fvecs = fvecs / fvecs.std(axis=0)
  cov = fvecs.T * fvecs / fvecs.shape[0]
  print c, \
    len(columns), \
    fvecs.shape[0], \
    -numpy.log(linalg.det(cov)) / numpy.log(cov.shape[0]), \
    numpy.sum(cov) / cov.shape[0] ** 2
