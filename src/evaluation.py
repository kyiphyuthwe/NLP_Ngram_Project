def calculate_sentence_probability(tokens, conditional_probs):
    """Calculate the probability of a sentence using bigram probabilities."""

    probability = 1.0

    for i in range(1, len(tokens)):
        bigram = (tokens[i - 1], tokens[i])

        if bigram in conditional_probs:
            probability *= conditional_probs[bigram]
        else:
            return 0.0

    return probability


def calculate_perplexity(tokens, conditional_probs):
    """Calculate perplexity of a sentence using bigram probabilities."""

    probability = calculate_sentence_probability(
        tokens,
        conditional_probs
    )

    if probability == 0:
        return float("inf")

    n = len(tokens) - 1

    return probability ** (-1 / n)