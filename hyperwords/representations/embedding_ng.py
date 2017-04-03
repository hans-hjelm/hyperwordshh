import heapq
import math

import numpy as np
from sklearn import preprocessing

from representations.matrix_serializer import load_count_vocabulary, load_vocabulary


class EmbeddingNg:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """
    
    def __init__(self, path, normalize=True, glen=5):
        self.m = np.load(path + '.npy')
        self.sz, self.ng_freqs = self.load_counts(path)
        self.glen = glen
        if normalize:
            self.normalize()
        self.dim = self.m.shape[1]
        self.wi, self.iw = load_vocabulary(path + '.vocab')

    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        self.m = self.m / norm[:, np.newaxis]

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
                    representation = self.m[self.wi[ngraph], :] * ngraph_si
                count += 1
        if count > 0:
            representation /= math.sqrt(sum(representation ** 2))
            return representation
        else:
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        return np.dot(self.represent(w1), self.represent(w2))


class SVDEmbeddingNg(EmbeddingNg):
    """
    SVD embeddings.
    Enables controlling the weighted exponent of the eigenvalue matrix (eig).
    Context embeddings can be created with "transpose".
    """
    
    def __init__(self, path, normalize=True, eig=0.0, glen=5, transpose=False):
        if transpose:
            ut = np.load(path + '.vt.npy')
            self.wi, self.iw = load_vocabulary(path + '.contexts.vocab')
        else:
            ut = np.load(path + '.ut.npy')
            self.wi, self.iw = load_vocabulary(path + '.words.vocab')
        s = np.load(path + '.s.npy')

        self.glen = glen
        self.sz, self.ng_freqs = self.load_counts(path)

        if eig == 0.0:
            self.m = ut.T
        elif eig == 1.0:
            self.m = s * ut.T
        else:
            self.m = np.power(s, eig) * ut.T

        self.dim = self.m.shape[1]

        if normalize:
            self.normalize()


class EnsembleEmbeddingNg(EmbeddingNg):
    """
    Adds the vectors of two distinct embeddings (of the same dimensionality) to create a new representation.
    Commonly used by adding the context embeddings to the word embeddings.
    """

    def __init__(self, emb1, emb2, normalize=False):
        """
        Assume emb1.dim == emb2.dim
        """
        self.dim = emb1.dim
        
        vocab1 = emb1.wi.keys()
        vocab2 = emb2.wi.keys()
        joint_vocab = list(vocab1 & vocab2)
        only_vocab1 = list(vocab1 - vocab2)
        only_vocab2 = list(vocab2 - vocab1)
        self.iw = joint_vocab + only_vocab1 + only_vocab2
        self.wi = dict([(w, i) for i, w in enumerate(self.iw)])

        m_joint = emb1.m[[emb1.wi[w] for w in joint_vocab]] + emb2.m[[emb2.wi[w] for w in joint_vocab]]
        m_only1 = emb1.m[[emb1.wi[w] for w in only_vocab1]]
        m_only2 = emb2.m[[emb2.wi[w] for w in only_vocab2]]
        self.m = np.vstack([m_joint, m_only1, m_only2])
        
        if normalize:
            self.normalize()


class DualEmbeddingWrapper:
    """
    Wraps word and context embeddings to allow investigation of first-order similarity.
    """

    def __init__(self, ew, ec):
        self.ew = ew
        self.ec = ec
    
    def closest_contexts(self, w, n=10):
        scores = self.ec.m.dot(self.ew.represent(w))
        pairs = zip(scores, self.ec.iw)[1:]
        return heapq.nlargest(n, pairs)
    
    def similarity_first_order(self, w, c):
        return self.ew.represent(w).dot(self.ec.represent(c))
