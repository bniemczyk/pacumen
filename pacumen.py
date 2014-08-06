import scapy.all as sc
import scipy.sparse as sparse
import numpy
import cPickle as pickle
import os
import oneclasstree

def make_feature_vectors_from_pcap(fname, bucketsize=10000, remove_empty_packets=True):
  cachefile = fname + ('.%d.fvec' % (bucketsize,))
  if os.path.exists(cachefile):
    try:
      with open(cachefile, 'rb') as f:
        rv = pickle.load(f)
        assert hasattr(rv, 'shape')
        return rv
    except:
      pass

  pkts = list(sc.rdpcap(fname))
  times = numpy.array([numpy.round(pkt.time, 3) * 1000 for pkt in pkts], dtype=numpy.long)
  times = numpy.sort(times)
  times = times - times[0]
  print 'TIMES: %s' % (times,)
  sizes = numpy.array([len(pkt) for pkt in pkts], dtype=numpy.long)
  buckets = times / bucketsize
  print 'BUCKETS: %s' % (buckets,)

  def _mk_sparse(bucket):
    szs = sizes[buckets == bucket]
    rv = sparse.lil_matrix((1,65536), dtype=numpy.long)
    for sz in szs:
      rv[0,sz] = rv[0,sz] + 1

    return rv

  svectors = [_mk_sparse(b) for b in range(buckets[-1] + 1)]
  print "SHAPES: %s" % ([s.shape for s in svectors],)

  rv = sparse.vstack(svectors).tocsr()

  # pick the smallest packet size and kill it!!!!! this gets rid of empty packets
  if remove_empty_packets:
    cols = rv.nonzero()[1]
    ep = cols.min()
    print 'assuming packet size %d is empty' % (ep,)
    rv[:,ep] = 0

  try:
    with open(cachefile, 'wb') as f:
      pickle.dump(rv, f, protocol=2)
  finally:
    return rv

def get_class_fvecs(classname, directory='tdata/train', remove_zero=True):
  fvs = []
  def _walk_fn(arg, dirname, fnames):
    fnames = ["%s/%s" % (dirname, f) for f in fnames if f[-5:] == '.pcap' and f[:len(classname)] == classname]
    fvs.extend([make_feature_vectors_from_pcap(f) for f in fnames])

  os.path.walk(directory, _walk_fn, None)
  fvs = sparse.vstack(fvs).tocsr()
  
  if remove_zero:
    nzrows = numpy.unique(fvs.nonzero()[0])
    fvs = fvs[nzrows,:]

  return fvs

def classify_pcap(classifier, pcap):
  fv = make_feature_vectors_from_pcap(pcap)
  result = classifier.classify(fv)
  result = oneclasstree.bayesian(result)[1]
  #print pcap, result, result > 0.5
  return result
