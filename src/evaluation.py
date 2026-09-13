from src.ngram_model import generate_ngrams


def calculate_sentence_probability(tokens, conditional_probs):
    """
    Calculate the probability of a sentence using a bigram model.

    <s>  = beginning of sentence
    </s> = end of sentence
    """

    if not tokens:
        return 0.0

    # Add sentence boundary tokens if necessary
    if tokens[0] != "<s>":
        tokens = ["<s>"] + tokens

    if tokens[-1] != "</s>":
        tokens = tokens + ["</s>"]

    probability = 1.0

    # Calculate:
    # P(w1 | <s>) *
    # P(w2 | w1) *
    # ...
    # P(</s> | wn)

    bigrams = generate_ngrams(tokens, 2)

    for bigram in bigrams:
        probability *= conditional_probs.get(bigram, 1e-6)

    return probability


def calculate_perplexity(tokens, conditional_probs):
    """
    Calculate perplexity using:

        PP(W) = P(W)^(-1/N)

    <s> is not counted in N.
    </s> is counted as a predicted token.
    """

    if not tokens:
        return float("inf")

    # Add sentence boundary tokens if necessary
    if tokens[0] != "<s>":
        tokens = ["<s>"] + tokens

    if tokens[-1] != "</s>":
        tokens = tokens + ["</s>"]

    probability = calculate_sentence_probability(
        tokens,
        conditional_probs
    )

    # Do not count <s>.
    # Count actual words + </s>.
    N = len(tokens) - 1

    if N <= 0:
        return float("inf")

    return probability ** (-1 / N)


