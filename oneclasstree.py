import numpy
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from symath.graph.directed import DirectedGraph
from symath import symbols
import pprint

def _toarray(mat):
  return mat.toarray() if hasattr(mat, 'toarray') else numpy.array(mat)

def bincount_oracle(X, columns=None):
  
  if columns == None:
    columns = numpy.unique(X.nonzero()[1])

  bins = {}

  for c in columns:
    bins[c] = numpy.bincount(_toarray(X[:,c].T)[0])
    bins[c] = bins[c] / numpy.sum(bins[c], dtype=numpy.double)

  def _rv(col, val):

    assert val >= 0

    if col not in bins:
      return 1.0 if val == 0 else 0.0

    return numpy.sum(bins[col][val:])

  return _rv

def _entropy_from_bincounts(bcs):
  if numpy.sum(bcs) == 0:
    return 0.0

  bcs = bcs / numpy.sum(bcs)
  nz = bcs > 0
  bcs[nz] = bcs[nz] * numpy.log(bcs[nz])
  
  return -numpy.sum(bcs)

_maximum_biclass_entropy = _entropy_from_bincounts(numpy.array([1.0, 1.0]))

def information_gain_ratio(X, oracle, column, cutoff):

  xlen = X.shape[0]

  coldata = _toarray(X[:,column].T)[0]

  # calculate >= entropy
  xcutofflen = coldata[coldata >= cutoff].shape[0]
  pcutofflen = numpy.int(numpy.round(xlen * oracle(column, cutoff)))
  geentropy = numpy.array([xcutofflen, pcutofflen], dtype=numpy.double)
  geentropy = _entropy_from_bincounts(geentropy)

  # calculate < entropy
  lessxcutofflen = xlen - xcutofflen
  lesspcutofflen = xlen - pcutofflen
  lentropy = numpy.array([lessxcutofflen, lesspcutofflen], dtype=numpy.double)
  lentropy = _entropy_from_bincounts(lentropy)

  # calculate the information gain
  Ht = _maximum_biclass_entropy
  geigain = (numpy.double(xcutofflen + pcutofflen) / (xlen * 2)) * geentropy
  ligain = (numpy.double(lessxcutofflen + lesspcutofflen) / (xlen * 2)) * lentropy
  igain = Ht - geigain - ligain

  # calculate the intrinsic value
  geiv = numpy.double(xcutofflen + pcutofflen) / (xlen *  2)
  if geiv > 0:
    geiv = geiv * numpy.log(geiv)

  liv = numpy.double(lessxcutofflen + lesspcutofflen) / (xlen * 2)
  if liv > 0:
    liv = liv * numpy.log(liv)

  iv = -geiv - liv

  ratio = numpy.round(igain / iv if iv > 0 else 0.0, 5)
  pospercentage = numpy.double(xcutofflen) / xlen
  negpercentage = oracle(column, cutoff)

  return numpy.array([ratio, pospercentage, negpercentage])

def _igain_filter(igains, igaincol, bias=0.0):

  assert igains.shape[0] > 0

  # get rid of total classifiers, we want weak classifiers
  igains = igains[igains[:,igaincol+1] < 1.0]

  if igains.shape[0] == 0:
    return igains

  igains = igains[igains[:,igaincol+1] > 0.0]

  if igains.shape[0] == 0:
    return igains

  # add in some regularization of sorts to choose filters that >= increase probability of positive class
  biased = igains[:,igaincol]

  # try to force an incremental search...
  # by taking a weighted geometric mean of:
  #     p(>=|pos) <-- regularization, try not to overfit: weight 1
  #     IGRatio <-- information gain: weight 1
  #     p(>=|pos)/p(>=|neg) <-- bias: weight bias

  biased = (biased * (igains[:,igaincol+1] / igains[:,igaincol+2]) ** bias * igains[:,igaincol+1]) ** (1.0 / (2.0 + bias))

  best_gain = numpy.max(biased)
  igains = igains[biased == best_gain, :]

  assert igains.shape[0] > 0

  if igains.shape[0] > 1:
    ps = numpy.abs(igains[:,igaincol+1] - 0.5) + numpy.abs(igains[:,igaincol+2] - 0.5)
    minp = numpy.min(ps)
    igains = igains[ps == minp, :]
    assert igains.shape[0] > 0

  return igains

def select_best_cutoff(X, oracle, column, bias=0.0):

  colvals = numpy.unique(_toarray(X[:,column].T)[0])

  igains = numpy.vstack([information_gain_ratio(X, oracle, column, cutoff) for cutoff in colvals])
  igains = numpy.hstack([numpy.array([colvals]).T, igains])
  igains = _igain_filter(igains, 1, bias=bias)
  
  return igains[numpy.random.randint(igains.shape[0])] if igains.shape[0] > 1 else igains

class NoBestColumn(Exception):
  def __init__(self):
    Exception.__init__(self, "No best column to select")

def select_best_column(X, oracle, bias=0.0):
  nzcols = numpy.unique(X.nonzero()[1])
  keepcols = []

  if nzcols.shape[0] == 0:
    raise NoBestColumn

  cutoffrows = []
  for col in nzcols:
    r = select_best_cutoff(X, oracle, col, bias=bias)
    if r.shape[0] == 0:
      continue
    cutoffrows.append(r)
    keepcols.append(col)

  if len(cutoffrows) == 0:
    raise NoBestColumn

  cutoffs = numpy.vstack(cutoffrows)
  cutoffs = numpy.hstack([numpy.array([keepcols]).T, cutoffs])
  cutoffs = _igain_filter(cutoffs, 2, bias=bias)
  
  return cutoffs[numpy.random.randint(cutoffs.shape[0])]

class OneClassTree(object):

  def __init__(self, X, oracle, target=0.6, maxdepth=5, g=None, eprobs=None, bias=0.0):

    if g == None:
      g = DirectedGraph()

    if eprobs == None:
      eprobs = numpy.array([1.0, 1.0])

    self.g = g
    self.eprobs = eprobs
    self.IS_LEAF = False
    self.column = None
    self.cutoff = None
    self._less_tree = None
    self._ge_tree = None

    probs = 0.5 * self.eprobs
    probs = probs / numpy.sum(probs)
    self.probs = probs

    met_target = numpy.all([p <= 1.0 - target or p >= target for p in probs])

    self.g.add_node(self)

    if met_target:
      self._make_leaf()

    elif X.shape[0] < 10:
      self._make_leaf()

    elif maxdepth <= 1:
      self._make_leaf()

    else:
      self._make_split(X, oracle, target, maxdepth, bias)

  def selected_columns(self):
    if self.IS_LEAF:
      return set([])
    else:
      left = self._less_tree.selected_columns()
      right = self._ge_tree.selected_columns()
      middle = set([self.column])
      return middle.union(left).union(right)

  def __getstate__(self):
    return (self.eprobs, self.IS_LEAF, self.column, self.cutoff, self._less_tree, self._ge_tree)

  def __setstate__(self, state):
    self.eprobs = state[0]
    self.IS_LEAF = state[1]
    self.column = state[2]
    self.cutoff = state[3]
    self._less_tree = state[4]
    self._ge_tree = state[5]
    self.g = DirectedGraph()
    self.g.add_node(self)

    self.probs = 0.5 * self.eprobs
    self.probs = self.probs / numpy.sum(self.probs)

    if self._less_tree != None:
      assert self._ge_tree != None

      self.g.union(self._less_tree.g)
      self.g.union(self._ge_tree.g)
      self._less_tree.g = self.g
      self._ge_tree.g = self.g
  
      less_edge = self.__symbolic_column_name__(self.column) < self.cutoff
      ge_edge = self.__symbolic_column_name__(self.column) >= self.cutoff
  
      self.g.connect(self, self._ge_tree, edgeValue=str(ge_edge))
      self.g.connect(self, self._less_tree, edgeValue=str(less_edge))


  def classify(self, x):
    assert x.shape[0] > 0

    if x.shape[0] > 1:
      rv = numpy.vstack([self.classify(x[i]) for i in range(x.shape[0])])

    elif self.IS_LEAF:
      rv = self.eprobs

    else:
      rv = self._less_tree.classify(x) if x[0, self.column] < self.cutoff else  self._ge_tree.classify(x)

    if len(rv.shape) == 1:
      rv = numpy.array([rv])

    return rv

  def _make_leaf(self):
    self.IS_LEAF = True

  def _make_split(self, X, oracle, target, maxdepth, bias):
    self.IS_LEAF = False

    try:
      best = select_best_column(X, oracle, bias=bias)
    except NoBestColumn:
      self._make_leaf()
      return

    self.column = int(best[0])
    self.cutoff = float(best[1])

    # get indexes for >= and <
    columndata = _toarray(X[:,self.column].T)[0]
    geidx = columndata >= self.cutoff
    lidx = geidx == False

    lX = X[lidx.nonzero()[0],:] if not numpy.all(lidx == False) else numpy.matrix((0, X.shape[1]))
    geX = X[geidx.nonzero()[0],:] if not numpy.all(geidx == False) else numpy.matrix((0, X.shape[1]))

    ge_eprobs = numpy.array([oracle(self.column, self.cutoff), float(geX.shape[0]) / X.shape[0]], dtype=numpy.double)
    l_eprobs = 1.0 - ge_eprobs
    ge_eprobs = ge_eprobs * self.eprobs
    l_eprobs = l_eprobs * self.eprobs

    # need to rebuild oracles to account for this split
    def loracle(col, cut):
      if col != self.column:
        return oracle(col, cut)

      if cut >= self.cutoff:
        return 0.0
      else:
        pless = (1.0 - oracle(col, cut)) / (1.0 - oracle(col, self.cutoff))
        return 1.0 - pless

    def georacle(col, cut):
      if col != self.column:
        return oracle(col, cut)

      if cut < self.cutoff:
        return 0.0
      else:
        origp = oracle(col, self.cutoff)
        newp = oracle(col, cut)
        assert origp >= newp

        lessorig = 1.0 - origp
        lessnew = 1.0 - newp

        less = (lessnew - lessorig) / (1.0 - lessorig)
        return 1.0 - less

    # generate trees
    self._ge_tree = self.__class__(geX, georacle, target, maxdepth-1, g=self.g, eprobs=ge_eprobs, bias=bias)
    self._less_tree = self.__class__(lX, loracle, target, maxdepth-1, g=self.g, eprobs=l_eprobs, bias=bias)

    less_edge = self.__symbolic_column_name__(self.column) < self.cutoff
    ge_edge = self.__symbolic_column_name__(self.column) >= self.cutoff

    self.g.connect(self, self._ge_tree, edgeValue=str(ge_edge))
    self.g.connect(self, self._less_tree, edgeValue=str(less_edge))

  def __symbolic_column_name__(self, colnum):
    x = symbols('x')
    return x(colnum)

  def __str__(self):
    return str(id(self)) + "\n" + str(numpy.vstack([self.probs, self.eprobs]))

  def __repr__(self):
    return str(self)


def bayesian(eproblist, probs=None, max_prob=0.99999, max_step=None):
  eproblist = numpy.array(eproblist)
  #print type(eproblist)
  if probs == None:
    probs = numpy.array([1.0 / eproblist.shape[1]] * eproblist.shape[1], dtype=numpy.double)

  for e in eproblist:
    oldprobs = probs
    probs = e * probs
    probs = probs / sum(probs)
    probs[probs > max_prob] = max_prob
    probs[probs < 1.0 - max_prob] = 1.0 - max_prob

    if max_step != None and numpy.abs(probs[0] - oldprobs[0]) > max_step:
      if probs[0] < oldprobs[0]:
        probs[0] = oldprobs[0] - max_step
        probs[1] = oldprobs[0] + max_step
      else:
        probs[0] = oldprobs[0] + max_step
        probs[1] = oldprobs[0] - max_step

  return probs
