from collections import Counter

from docopt import docopt

from representations.matrix_serializer import save_count_vocabulary


def main():
    args = docopt("""
    Usage:
        counts2pmi.py <counts>
    """)
    
    counts_path = args['<counts>']

    words = Counter()
    contexts = Counter()
    with open(counts_path) as f:
        for line in f:
            if line.__contains__('\t'):
                count, word, context = line.strip('\n').split('\t')
            else:
                count, word, context = line.strip().split()
            count = int(count.strip())
            words[word] += count
            contexts[context] += count

    words = sorted(words.items(), key = lambda wfreq: wfreq[1], reverse=True)
    contexts = sorted(contexts.items(), key=lambda wfreq: wfreq[1], reverse=True)

    save_count_vocabulary(counts_path + '.words.vocab', words)
    save_count_vocabulary(counts_path + '.contexts.vocab', contexts)


if __name__ == '__main__':
    main()
