#!/usr/bin/env python

import setuptools

_depends = '''
scipy
numpy
symath
pydot
scapy
'''

setuptools.setup( \
  name='pacumen', \
  version='git', \
  description='ML protocol identification based on packet sizes', \
  author='Brandon Niemczyk', \
  author_email='brandon.niemczyk@gmail.com', \
  url='http://github.com/bniemczyk/pacumen', \
  py_modules=['oneclasstree', 'pacumen', 'multinormal'], \
  scripts=['pacumen_train.py', 'pacumen_classify_pcap.py', 'pacumen_visualize.py', 'pacumen_timeseries.py', 'train_all.py', 'xval.py'], \
  test_suite='tests', \
  license='BSD', \
  install_requires=_depends, \
  zip_safe=True, \
  classifiers = [ \
    'Development Status :: 3 - Alpha', \
    'Intended Audience :: Developers', \
    'Intended Audience :: Science/Research', \
    'License :: OSI Approved :: BSD License', \
    'Topic :: Scientific/Engineering :: Mathematics' \
    ]
  )
