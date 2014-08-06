#!/usr/bin/env python
import os
import numpy
import sys
import cPickle as pickle
import pacumen
import matplotlib.pyplot as plt

THRESH = 0.5

if len(sys.argv) > 1:
    classfiles = sys.argv[1:]
else:
    classfiles = []
    [classfiles.extend(["%s%s" % (root, f) for f in files]) for root,dirs,files in os.walk('classifiers/')]

def is_labeled_class(classname, xvalfile):
  fname = xvalfile.split('/')[-1]
  return classname == fname[:len(classname)]

def xval_classfile(classfile):
  print classfile
  with open(classfile, 'rb') as f:
    classifier = pickle.load(f)

  classname = classfile.split('/')[-1]
  xvalfiles = []
  [xvalfiles.extend(["%s/%s" % (root, f) for f in files if f[-5:] == '.pcap']) for root, dirs, files in os.walk('tdata/xval')]

  TP = 0
  FP = 0
  TN = 0
  FN = 0

  true_class_probs = []
  neg_class_probs = []

  for xf in xvalfiles:

    res = pacumen.classify_pcap(classifier, xf)

    if is_labeled_class(classname, xf):
      true_class_probs.append(res)
    else:
      neg_class_probs.append(res)

    #print xf, res

    if res > THRESH:
      if is_labeled_class(classname, xf):
        TP += 1
      else:
        FP += 1
    else:
      if is_labeled_class(classname, xf):
        FN += 1
      else:
        TN += 1

  accuracy = float(TP + TN) / (TP + TN + FP + FN)
  precision = float(TP) / (TP + FP)
  recall = float(TP) / (TP + FN)
  if precision + recall != 0:
    fscore = 2 * precision * recall / (precision + recall)
  else:
    fscore = numpy.nan

  print '''
  True Positives:  %d
  True Negatives:  %d
  False Positives: %d
  False Negatives: %d

  Accuracy: %f
  Precision: %f
  Recall: %f
  F-score: %f
  ''' % (TP, TN, FP, FN, accuracy, precision, recall, fscore)

  plt.figure()
  plt.hist(true_class_probs, color='blue')
  plt.hist(neg_class_probs, color='red')
  plt.title('%s' % (classname))
  plt.savefig('histograms/%s.png' % (classname))

[xval_classfile(cf) for cf in classfiles]
