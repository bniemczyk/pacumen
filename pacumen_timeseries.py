import automatamm
import particle
import scapy.all as sc
import pacumen
import os
import cPickle as pickle

def make_timeseries_from_pcap(fname):
  cachefile = fname + '.timeseries'

  try:
    with open(cachefile, 'rb') as f:
      rv = pickle.load(f)
      assert hasattr(rv, 'shape')
      assert len(rv.shape) == 1
      assert rv.dtype == numpy.long
      return rv
  except:

    pkts = list(scapy.rdpcap(fname))
    times = numpy.array([numpy.round(pkt.time, 3) * 1000 for pkt in pkts], dtype=numpy.long)
    timeseries = times[1:] - times[:-1]

    try:
      with open(cachefile, 'wb') as f:
        pickle.dump(timeseries, f, protocol=2)
    except:
      print 'could not save cache of timeseries in %s' % (cachefile,)

    return timeseries

def find_pcaps(direc='tdata'):

  rv = []
  for root,_,files in os.walk(direc):
    rv = rv + [root + '/' + f for f in files if f[-5:] == '.pcap']

  return rv

def make_timeseries_gram(pcaps):
  l = len(pcaps)
  rv = numpy.eye(l,dtype=numpy.double)
  for i in range(len(pcaps)):
    for j in range(i+1, len(pcaps)):
      a = make_timeseries_from_pcap(pcaps[i])
      b = make_timeseries_from_pcap(pcaps[j])
      rv[i,j] = automatamm.timeseries_kernel(a,b,smooth=20,normalize=False)
      rv[j,i] = rv[i,j]
  return rv
