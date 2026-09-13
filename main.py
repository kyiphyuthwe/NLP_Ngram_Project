from src.preprocessing import preprocess_text
from src.ngram_model import (
    generate_ngrams,
    calculate_probabilities,
    calculate_conditional_probabilities,
    predict_next_word
)
from src.evaluation import (
    calculate_sentence_probability,
    calculate_perplexity
)

# Load default corpus
with open("corpora/default.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokens = preprocess_text(text)

print("Corpus Words:", len(tokens))
print()

# ----------------------------
# Unigram
# ----------------------------
unigrams = generate_ngrams(tokens, 1)
unigram_counts, unigram_probs = calculate_probabilities(unigrams)

print("=== UNIGRAM ===")
for ngram, count in list(unigram_counts.items())[:10]:
    print(ngram, "Count:", count, "Probability:", unigram_probs[ngram])

print()

# ----------------------------
# Bigram
# ----------------------------
bigrams = generate_ngrams(tokens, 2)
bigram_counts, bigram_probs = calculate_probabilities(bigrams)

print("=== BIGRAM ===")
for ngram, count in list(bigram_counts.items())[:10]:
    print(ngram, "Count:", count, "Probability:", bigram_probs[ngram])

print()

# ----------------------------
# Trigram
# ----------------------------
trigrams = generate_ngrams(tokens, 3)
trigram_counts, trigram_probs = calculate_probabilities(trigrams)

print("=== TRIGRAM ===")
for ngram, count in list(trigram_counts.items())[:10]:
    print(ngram, "Count:", count, "Probability:", trigram_probs[ngram])

print()

# ----------------------------
# Conditional Probabilities
# ----------------------------
conditional_probs = calculate_conditional_probabilities(bigrams)

print("=== CONDITIONAL BIGRAM PROBABILITIES ===")
for ngram, probability in list(conditional_probs.items())[:10]:
    print(f"P({ngram[1]} | {ngram[0]}) = {probability:.4f}")

print()

# ----------------------------
# Top Predictions
# ----------------------------
context = ("machine",)

predictions = predict_next_word(context, conditional_probs)

print("=== TOP PREDICTIONS ===")
for i, (word, prob) in enumerate(predictions[:3], 1):
    print(f"{i}. {word} ({prob:.4f})")

# ----------------------------
# Sentence Probability
# ----------------------------
sentence = "machine learning is useful"
sentence_tokens = preprocess_text(sentence)

sentence_probability = calculate_sentence_probability(
    sentence_tokens,
    conditional_probs
)

print()
print("=== SENTENCE PROBABILITY ===")
print(f"Sentence: {sentence}")
print(f"Probability: {sentence_probability:.6f}")

# ----------------------------
# Perplexity
# ----------------------------
perplexity = calculate_perplexity(
    sentence_tokens,
    conditional_probs
)

print()
print("=== PERPLEXITY ===")
print(f"Sentence: {sentence}")
print(f"Perplexity: {perplexity:.4f}")