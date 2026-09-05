from collections import Counter


def generate_ngrams(tokens, n):
    """Generate n-grams from a list of tokens."""
    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def calculate_probabilities(ngrams):
    """Calculate probability of each n-gram."""
    counts = Counter(ngrams)
    total = sum(counts.values())

    probabilities = {
        ngram: count / total
        for ngram, count in counts.items()
    }

    return counts, probabilities

def calculate_conditional_probabilities(ngrams):
    """Calculate conditional probabilities for n-grams."""

    n = len(ngrams[0])

    ngram_counts = Counter(ngrams)
    context_counts = Counter(
        ngram[:-1] for ngram in ngrams
    )

    probabilities = {}

    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        probabilities[ngram] = count / context_counts[context]

    return probabilities