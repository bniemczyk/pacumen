import scipy
import scipy.linalg as dlinalg
import scipy.sparse as sparse
import numpy
import os
import pacumen

class MultiNormalLikelihood(object):

  def __init__(self, pos, neg, k=5):
    self.k = k
    pos = pos.T
    neg = neg.T

    totals = pos.sum(axis=1)
    popular = numpy.argsort(totals, axis=0)[::-1,:][1:1+k,:]
    popular = numpy.array(popular.T)[0]
    self.popular = popular

    pos = pos[popular,:].todense()
    neg = neg[popular,:].todense()

    self.posmu = pos.mean(axis=1)
    self.negmu = neg.mean(axis=1)

    p = pos - self.posmu
    n = neg - self.negmu

    self.pcov = p * p.T / p.shape[1]
    self.ncov = n * n.T / n.shape[1]

    self.pdet = dlinalg.det(self.pcov)
    self.ndet = dlinalg.det(self.ncov)

    assert self.pdet != 0
    assert self.ndet != 0

    self.picov = dlinalg.inv(self.pcov)
    self.nicov = dlinalg.inv(self.ncov)

  @staticmethod
  def _likelihood(x, mu, det, cov, icov, popular, k):
    x = x.T
    x = x[popular,:].todense()
    x = x - mu
    rv = -0.5 * numpy.log(det)
    rv -= 0.5 * x.T * icov * x
    rv -= (k / 2.0) * numpy.log(2 * numpy.pi)
    assert rv.shape == (1,1)
    return numpy.exp(rv[0,0])

  def classify(self, x):
    posrv = [self._likelihood(x[i,:], self.posmu, self.pdet, self.pcov, self.picov, self.popular, self.k) for i in range(x.shape[0])]
    negrv = [self._likelihood(x[i,:], self.negmu, self.ndet, self.ncov, self.nicov, self.popular, self.k) for i in range(x.shape[0])]

    rv = numpy.matrix([negrv, posrv]).T
    return rv
