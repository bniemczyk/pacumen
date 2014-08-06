#!/usr/bin/env python

import os
import sys
import regex

pattern = None

if len(sys.argv) > 1:
  classnames = [sys.argv[1]]
  if len(sys.argv) >= 3:
    pattern = '.*' + sys.argv[2] + '.*'

else:
  classnames = os.popen('ls tdata/train | perl -pe \'s/^(.*?)_00.*/$1/g\' | uniq', 'r').readlines()
  classnames = map(lambda x: x.strip(), classnames)

for cn in classnames:
  if pattern == None:
    posfiles = os.popen('ls tdata/train/%s*.pcap' % (cn,), 'r').readlines()
    posfiles = map(lambda x: '-T ' + x.strip(), posfiles)
   
    negfiles = os.popen('ls tdata/train/*.pcap | grep -Pv \'^%s\'' % (cn,), 'r').readlines()
    negfiles = map(lambda x: '-N ' + x.strip(), negfiles)

  else:
    pattern = regex.compile(pattern)
    files = os.popen('ls tdata/train/*.pcap', 'r').readlines()
    files = [x.strip() for x in files]

    posfiles = ['-T ' + x for x in files if pattern.match(x) != None]
    negfiles = ['-N ' + x for x in files if pattern.match(x) == None]

  outfile = 'classifiers/%s' % (cn,)
 
  cmd = './pacumen_train.py -M 2 ' + ' '.join(posfiles + negfiles) + ' ' + outfile
  print cmd
  os.system(cmd)
