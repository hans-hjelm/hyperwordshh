from representations.embedding import SVDEmbedding, EnsembleEmbedding, Embedding
from representations.embedding_ng import SVDEmbeddingNg
from representations.explicit import PositiveExplicit
from representations.explicit_ng import PositiveExplicitNg


def create_representation(args):
    rep_type = args['<representation>']
    path = args['<representation_path>']
    neg = int(args['--neg'])
    w_c = args['--w+c']
    eig = float(args['--eig'])
    if args['--len']:
        glen = int(args['--len'])
    
    if rep_type == 'PPMI':
        if w_c:
            raise Exception('w+c is not implemented for PPMI.')
        else:
            return PositiveExplicit(path, True, neg)
        
    elif rep_type == 'SVD':
        if w_c:
            return EnsembleEmbedding(SVDEmbedding(path, False, eig, False), SVDEmbedding(path, False, eig, True), True)
        else:
            return SVDEmbedding(path, True, eig)

    elif rep_type == 'PPMIng':
        return PositiveExplicitNg(path, True, neg, glen)

    elif rep_type == 'SVDng':
        return SVDEmbeddingNg(path, True, eig, glen)
        
    else:
        if w_c:
            return EnsembleEmbedding(Embedding(path + '.words', False), Embedding(path + '.contexts', False), True)
        else:
            return Embedding(path + '.words', True)
