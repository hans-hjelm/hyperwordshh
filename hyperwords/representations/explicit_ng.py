import heapq
import math

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from sklearn import preprocessing

from representations.matrix_serializer import load_count_vocabulary, load_vocabulary, load_matrix


class ExplicitNg:
    """
    Class for explicit representations using ngraphs. Assumes that the serialized input is e^PMI.
    """
    
    def __init__(self, path, normalize=True, glen=5):
        self.wi, self.iw = load_vocabulary(path + '.words.vocab')
        self.ci, self.ic = load_vocabulary(path + '.contexts.vocab')
        self.sz, self.ng_freqs = self.load_counts(path)
        self.m = load_matrix(path)
        self.m.data = np.log(self.m.data)
        self.normal = normalize
        self.glen = glen
        if normalize:
            self.normalize()
    
    def normalize(self):
        m2 = self.m.copy()
        m2.data **= 2
        norm = np.reciprocal(np.sqrt(np.array(m2.sum(axis=1))[:, 0]))
        normalizer = dok_matrix((len(norm), len(norm)))
        normalizer.setdiag(norm)
        self.m = normalizer.tocsr().dot(self.m)

    def load_counts(self, path):
        count_path = path[:path.rfind('/') + 1] + 'counts.words.vocab'
        ng_freqs = load_count_vocabulary(count_path)
        sz = sum(int(v) for v in ng_freqs.values())
        return sz, ng_freqs

    def represent(self, w):
        representation = None
        count = 0
        for i in range(0, len(w) + 1 - self.glen):
            ngraph = w[i:i + self.glen]
            if ngraph in self.wi:
                ngraph_si = math.log2(1/(int(self.ng_freqs.get(ngraph))/self.sz))
                if count > 0:
                    representation += self.m[self.wi[ngraph], :] * ngraph_si
                else:
                    representation  = self.m[self.wi[ngraph], :] * ngraph_si
                count += 1
        if count > 0:
            representation = preprocessing.normalize(representation, norm='l2')
            return representation
        else:
            return csr_matrix((1, len(self.ic)))
    
    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        return self.represent(w1).dot(self.represent(w2).T)[0, 0]
    

class PositiveExplicitNg(ExplicitNg):
    """
    Positive PMI (PPMI) with negative sampling (neg) for ngraphs.
    Negative samples shift the PMI matrix before truncation.
    """
    
    def __init__(self, path, normalize=True, neg=1, glen=5):
        ExplicitNg.__init__(self, path, False, glen)
        self.m.data -= np.log(neg)
        self.m.data[self.m.data < 0] = 0
        self.m.eliminate_zeros()
        if normalize:
            self.normalize()
