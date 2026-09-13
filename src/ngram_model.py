from collections import Counter

def generate_ngrams(tokens, n):
    return [
        tuple(tokens[i:i+n])
        for i in range(len(tokens)-n+1)
    ]

def calculate_probabilities(ngrams):
    counts = Counter(ngrams)
    total = sum(counts.values())

    probs = {
        ngram: count/total
        for ngram, count in counts.items()
    }

    return counts, probs

def calculate_conditional_probabilities(ngrams):

    if not ngrams:
        return {}

    ngram_counts = Counter(ngrams)

    context_counts = Counter(
        ngram[:-1]
        for ngram in ngrams
    )

    probs = {}

    for ngram, count in ngram_counts.items():
        probs[ngram] = count/context_counts[ngram[:-1]]

    return probs

def predict_next_word(context, conditional_probs):
    candidates = []

    for ngram, probability in conditional_probs.items():
        if ngram[:-1] == context:
            candidates.append((ngram[-1], probability))

    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates