from src.ngram_model import generate_ngrams

def calculate_sentence_probability(tokens, conditional_probs):
    """
    Calculate sentence probability using a bigram model.
    """

    if len(tokens) == 0:
        return 0

    # Add boundary tokens if the sentence doesn't already contain them.
    if tokens[0] != "<s>":
        tokens = ["<s>"] + tokens + ["</s>"]

    probability = 1.0

    # First word probability: P(w1 | <s>)
    first_bigram = (tokens[0], tokens[1])
    probability *= conditional_probs.get(first_bigram, 1e-6)

    # Remaining bigrams
    for bigram in generate_ngrams(tokens[1:], 2):
        probability *= conditional_probs.get(bigram, 1e-6)

    return probability


def calculate_perplexity(tokens, conditional_probs):
    """
    Universal perplexity formula:
    PP = P(W)^(-1/N)
    """

    if len(tokens) == 0:
        return float("inf")

    probability = calculate_sentence_probability(tokens, conditional_probs)

    N = len(tokens)

    return probability ** (-1 / N)
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
    # Count the actual words + </s>.
    N = len(tokens) - 1

    if N <= 0:
        return float("inf")

    return probability ** (-1 / N)
