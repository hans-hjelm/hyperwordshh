import heapq

from scipy.sparse import dok_matrix, csr_matrix
import numpy as np

from representations.matrix_serializer import load_vocabulary, load_matrix


class ExplicitNg:
    """
    Class for explicit representations using ngraphs. Assumes that the serialized input is e^PMI.
    """
    
    def __init__(self, path, normalize=True, glen=5):
        self.wi, self.iw = load_vocabulary(path + '.words.vocab')
        self.ci, self.ic = load_vocabulary(path + '.contexts.vocab')
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
    
    def represent(self, w):
        # TODO: normalize to unit length and weight by self information
        representation = None
        count = 0
        for i in range(0, len(w) + 1 - self.glen):
            ngraph = w[i:i + self.glen]
            if ngraph in self.wi:
                if count > 0:
                    representation += self.m[self.wi[ngraph], :]
                else:
                    representation  = self.m[self.wi[ngraph], :]
                count += 1
        if count > 0:
            representation /= count
            return representation
        else:
            return csr_matrix((1, len(self.ic)))
    
    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        return self.represent(w1).dot(self.represent(w2).T)[0, 0]
    
    def closest_contexts(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.represent(w)
        return heapq.nlargest(n, zip(scores.data, [self.ic[i] for i in scores.indices]))
    
    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w).T).T.tocsr()
        return heapq.nlargest(n, zip(scores.data, [self.iw[i] for i in scores.indices]))


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
