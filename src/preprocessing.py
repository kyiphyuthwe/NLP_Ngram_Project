import re

def preprocess_text(text):
    """
    Lowercase, remove punctuation, split into words,
    and add sentence boundary tokens.
    """

    text = text.lower()

    # Keep only letters, spaces and sentence-ending punctuation.
    text = re.sub(r"[^a-z.!?\s]", "", text)

    sentences = re.split(r"[.!?]+", text)

    tokens = []

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence:
            tokens.append("<s>")
            tokens.extend(sentence.split())
            tokens.append("</s>")

    return tokens