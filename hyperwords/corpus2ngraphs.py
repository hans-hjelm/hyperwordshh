from collections import Counter
from math import sqrt
from random import Random
import re

from docopt import docopt

pattern = re.compile('[^\w\s]+', re.UNICODE)

def main():
    args = docopt("""
    Usage:
        corpus2ngraphs.py [options] <corpus>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size in characters [default: 30]
        --len NUM    Length of ngraph
        --pos        Positional contexts
        --dyn        Dynamic context windows
        --sub NUM    Subsampling threshold [default: 0]
        --del        Delete out-of-vocabulary and subsampled placeholders
    """)
    
    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])
    glen = int(args['--len'])
    pos = args['--pos']
    dyn = args['--dyn']
    subsample = float(args['--sub'])
    sub = subsample != 0
    d3l = args['--del']

    vocab = read_vocab(corpus_file, thr, glen)
    corpus_size = sum(vocab.values())
    
    subsample *= corpus_size
    subsampler = dict([(word, 1 - sqrt(subsample / count)) for word, count in vocab.items() if count > subsample])
    
    rnd = Random(17)
    ngraphs = list()
    with open(corpus_file) as f: 
        for line in f:
            
            line = pattern.sub('', line)
            line = ' ' + line.lower().strip() + ' '
            ngraphs.clear()
            for i in range(0, len(line) + 1 - glen):
                ngraphs.append(line[i:i + glen])
            tokens = [t if t in vocab else None for t in ngraphs]
            if sub:
                tokens = [t if t not in subsampler or rnd.random() > subsampler[t] else None for t in tokens]
            if d3l:
                tokens = [t for t in tokens if t is not None]
            
            len_tokens = len(tokens)
            
            for i, tok in enumerate(tokens):
                if tok is not None:
                    if dyn:
                        dynamic_window = rnd.randint(1, win)
                    else:
                        dynamic_window = win
                    start = i - dynamic_window
                    if start < 0:
                        start = 0
                    end = i + dynamic_window + 1
                    if end > len_tokens:
                        end = len_tokens
                    
                    if pos:
                        output = '\n\t'.join([row for row in [tok + '\t' + tokens[j] + '_' + str(j - i) for j in range(start, end) if j != i and tokens[j] is not None] if len(row) > 0])
                    else:
                        output = '\n\t'.join([row for row in [tok + '\t' + tokens[j] for j in range(start, end) if j != i and tokens[j] is not None] if len(row) > 0])
                    if len(output) > 0:
                        print('\t' + output)


def read_vocab(corpus_file, thr, glen):
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            line = pattern.sub('', line)
            line = ' ' + line.lower().strip() + ' '
            if len(line) < glen:
                continue
            for i in range(0, len(line) + 1 - glen):
                vocab.update([line[i:i + glen]])
    return dict([(token, count) for token, count in vocab.items() if count >= thr])


if __name__ == '__main__':
    main()
