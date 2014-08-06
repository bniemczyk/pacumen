#!/usr/bin/env python

import pacumen
import getopt
import sys
import scipy.sparse as sparse
import oneclasstree
import cPickle as pickle
import numpy
import multinormal

def print_help():
  print '''
  train a classifier for a protocol over an encryptd tunnel given pcaps
  you can specify multiple pcaps that belong to the target protocol and
  multiple pcaps that belong do not belong to the target protocol
  at least one of each must be supplied

  an output filename must also be supplied
  '''

  print 'usage: %s [-T target] [-N nontarget] [-B bias] <outputfilename>' % (sys.argv[0])

  print '''
  -T target: specify a pcap with the target protocol, may be used multiple times
  -N nontarget: specifiy a pcap without the target protocol, may be used multiple times
  -B float: bias towards >= indicates positive class (default: auto), -bias may be used for the opposite
  (optional) -D max depth: maximum depth of decision tree (default: 9)
  (optional) -M minimum bias: minimum bias to use with -B auto
  '''

options,remainder = getopt.getopt(sys.argv[1:], 'T:N:hD:B:M:Gk:')

targets = set()
nontargets = set()
maxdepth = 9
bias = 'auto'
gaussian = False

maxbias = 8.0
minbias = 0.0
k = 5

for opt,arg in options:

  if opt == '-T':
    targets.add(arg)
  elif opt == '-N':
    nontargets.add(arg)
  elif opt == '-h':
    print_help()
    exit()
  elif opt == '-D':
    maxdepth = int(arg)
  elif opt == '-B':
    bias = float(arg) if arg.lower().strip() != 'auto' else 'auto'
  elif opt == '-M':
    minbias = float(arg)
  elif opt == '-G':
    gaussian = True
  elif opt == '-k':
    k = int(arg)

if len(remainder) != 1 or len(targets) == 0 or len(nontargets) == 0:
  print_help()
  exit()

outfilename = remainder[0]


print 'reading pcaps'

ntmat = sparse.vstack([pacumen.make_feature_vectors_from_pcap(pcap) for pcap in nontargets]).tocsr()
tmat = sparse.vstack([pacumen.make_feature_vectors_from_pcap(pcap) for pcap in targets]).tocsr()

classifier = None

print 'have %d rows of target data and %d rows of non-target data' % (tmat.shape[0], ntmat.shape[0])

all_data = sparse.vstack([ntmat, tmat]).tocsr()
oracle = oneclasstree.bincount_oracle(all_data)

best = None

if gaussian:
  print 'using mutlivariate gaussian likelihood function'
  classifier = multinormal.MultiNormalLikelihood(tmat, all_data, k=k)

elif bias != 'auto':
  print 'training'
  classifier = oneclasstree.OneClassTree(tmat, oracle, maxdepth=maxdepth, target=0.9, bias=bias)

else: # need to try many biases
  zerovector = sparse.csr_matrix((1, tmat.shape[1]), dtype=tmat.dtype)
  maxiters = 22

  # try it with no bias and see if that works
  classifier = oneclasstree.OneClassTree(tmat, oracle, maxdepth=maxdepth, target=0.9, bias=minbias)
  zvp = classifier.classify(zerovector)[0]
  zvp /= 2
  zvp /= numpy.sum(zvp)
  zvp = zvp[1]

  if zvp >= 0.5:
    # uh oh, gotta find a bias
    for i in range(maxiters):
      current = numpy.mean([maxbias, minbias])
  
      classifier = oneclasstree.OneClassTree(tmat, oracle, maxdepth=maxdepth, target=0.9, bias=current)
  
      # calculate P(POSITIVE|ZEROVECTOR)
      zvp = classifier.classify(zerovector)[0]
      zvp /= 2
      zvp /= numpy.sum(zvp)
      zvp = zvp[1]
  
      print 'bias %f, p(positive|zerovector) = %f' % (current, zvp)
  
      if zvp < 0.5: # we may be able to use a smaller bias
        maxbias = current
        if best == None or current < best[0]:
          best = (current,classifier)
      else: # we need a larger bias
        minbias = current
  
if best != None:
  print 'using bias %f' % (best[0])
  classifier = best[1]

print 'writing classifier to file'
with open(outfilename, 'wb') as f:
  pickle.dump(classifier, f, protocol=2)

if hasattr(classifier, 'selected_columns'):
  print 'Used packet sizes: %s' % (classifier.selected_columns(),)
