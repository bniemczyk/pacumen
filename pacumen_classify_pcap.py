#!/usr/bin/env python

import pacumen
import getopt
import sys
import scipy.sparse as sparse
import oneclasstree
import cPickle as pickle
import numpy
import matplotlib.pyplot as plt

def print_help():
  print '''
  run a classifier against a pcap and print the probability that it contains the classified protocol

  usage: %s -C classifier <pcap files>

  ''' % (sys.argv[0])

options,remainder = getopt.getopt(sys.argv[1:], 'C:V')
classifier = None
visualize = False

for opt,arg in options:
  if opt == '-C':
    classifier = arg
  elif opt == '-V':
    visualize = True
    if plt.get_backend().lower() == 'agg':
      plt.switch_backend("Qt4Agg")

if classifier == None or len(remainder) < 1:
  print_help()
  exit()

try:
  with open(classifier, 'rb') as f:
    classifier = pickle.load(f)
    assert hasattr(classifier, 'classify')
except:
  print "could not load classifier"
  print_help()
  exit()

def _plot(eprobs):
  print 'plotting'
  probs = numpy.array([0.5, 0.5])
  lst = [oneclasstree.bayesian(eprobs[:x,:])[1] for x in range(1, eprobs.shape[0]+1)]
  plt.figure()
  plt.plot(lst)
  plt.show()


for pcap in remainder:
  X = pacumen.make_feature_vectors_from_pcap(pcap)
  eprobs = classifier.classify(X)
  if visualize:
    _plot(eprobs)
  eprobs = oneclasstree.bayesian(eprobs)
  print '%f  %s' % (eprobs[1], pcap)
