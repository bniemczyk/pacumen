#!/usr/bin/env python

import oneclasstree
import cPickle as pickle
import sys

if len(sys.argv) != 2:
  print '''
  usage %s classifier-file
  ''' % sys.argv[0]
  exit()

with open(sys.argv[1], 'rb') as f:
  classifier = pickle.load(f)

classifier.g.visualize()
