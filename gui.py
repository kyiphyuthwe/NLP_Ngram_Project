import os
import shutil
import customtkinter as ctk
from tkinter import filedialog

from nltk.corpus import brown, gutenberg, reuters, webtext

from src.preprocessing import preprocess_text
from src.ngram_model import (
    generate_ngrams,
    calculate_conditional_probabilities,
    predict_next_word
)
from src.evaluation import (
    calculate_sentence_probability,
    calculate_perplexity
)

# ----------------------------
# App Settings
# ----------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("N-Gram Language Model")
app.geometry("980x640")
app.resizable(False, False)

# ----------------------------
# Corpus Folder
# ----------------------------
CORPUS_FOLDER = "corpora"

if not os.path.exists(CORPUS_FOLDER):
    os.makedirs(CORPUS_FOLDER)

# ----------------------------
# Global Model Variables
# ----------------------------
corpus_tokens = []
conditional_probs = {}

# ----------------------------
# NLTK Corpus Names
# ----------------------------
NLTK_CORPORA = [
    "Brown",
    "Gutenberg",
    "Reuters",
    "Webtext"
]

# ----------------------------
# Corpus Functions
# ----------------------------
def get_corpus_list():
    custom_files = [
        f for f in os.listdir(CORPUS_FOLDER)
        if f.endswith(".txt")
    ]

    custom_files.sort()

    return custom_files + NLTK_CORPORA


def get_nltk_tokens(name):
    """
    Load tokens from an NLTK corpus while preserving
    sentence boundaries.
    """

    if name == "Brown":
        sentences = brown.sents()

    elif name == "Gutenberg":
        sentences = gutenberg.sents()

    elif name == "Reuters":
        sentences = reuters.sents()

    elif name == "Webtext":
        sentences = webtext.sents()

    else:
        return []

    tokens = []

    for sentence in sentences:
        cleaned_sentence = []

        for word in sentence:
            word = word.lower()

            # Keep alphabetic words only
            if word.isalpha():
                cleaned_sentence.append(word)

        if cleaned_sentence:
            tokens.append("<s>")
            tokens.extend(cleaned_sentence)
            tokens.append("</s>")

    return tokens


def load_corpus(name):
    global corpus_tokens, conditional_probs

    status.configure(
        text="● Loading corpus...",
        text_color="#F59E0B"
    )

    app.update_idletasks()

    # ----------------------------
    # Custom TXT Corpus
    # ----------------------------
    if name.endswith(".txt"):

        path = os.path.join(CORPUS_FOLDER, name)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        corpus_tokens = preprocess_text(text)

    # ----------------------------
    # NLTK Corpus
    # ----------------------------
    else:

        corpus_tokens = get_nltk_tokens(name)

    # ----------------------------
    # Build Bigram Model
    # ----------------------------
    bigrams = generate_ngrams(corpus_tokens, 2)

    conditional_probs = calculate_conditional_probabilities(
        bigrams
    )

    # ----------------------------
    # Update Statistics
    # ----------------------------
    corpus_label.configure(
        text=f"Corpus Size: {len(corpus_tokens)} tokens"
    )

    vocab_label.configure(
        text=f"Vocabulary: {len(set(corpus_tokens))} words"
    )

    status.configure(
        text="● Model Ready",
        text_color="#16A34A"
    )

    result_box.configure(
        text=f"Loaded corpus:\n{name}"
    )


def upload_corpus():
    path = filedialog.askopenfilename(
        title="Select Corpus",
        filetypes=[("Text Files", "*.txt")]
    )

    if not path:
        return

    filename = os.path.basename(path)
    destination = os.path.join(CORPUS_FOLDER, filename)

    shutil.copy(path, destination)

    corpus_menu.configure(
        values=get_corpus_list()
    )

    corpus_menu.set(filename)

    load_corpus(filename)


# ----------------------------
# Helper Functions
# ----------------------------
def get_input():
    text = input_box.get().strip()

    if text == "":
        result_box.configure(
            text="Please enter a word or sentence."
        )
        return None

    return text


# ----------------------------
# Prediction
# ----------------------------
def predict():
    text = get_input()

    if text is None:
        return

    # Preprocess user input
    context_tokens = [
        token
        for token in preprocess_text(text)
        if token not in ("<s>", "</s>")
    ]

    if len(context_tokens) == 0:
        result_box.configure(
            text="Please enter valid text."
        )
        return

    context = tuple(context_tokens)

    # Context length + 1 = N-gram size
    n = len(context) + 1

    ngrams = generate_ngrams(
        corpus_tokens,
        n
    )

    probs = calculate_conditional_probabilities(
        ngrams
    )

    predictions = predict_next_word(
        context,
        probs
    )

    if not predictions:
        result_box.configure(
            text=f"No prediction found for:\n\n{text}"
        )
        return

    output = (
        f"Input:\n{text}\n\n"
        f"Top Predictions\n\n"
    )

    for i, (word, probability) in enumerate(
        predictions[:3],
        start=1
    ):
        output += (
            f"{i}. {word} "
            f"({probability:.4f})\n"
        )

    result_box.configure(
        text=output
    )


# ----------------------------
# Sentence Probability
# ----------------------------
def sentence_probability():

    sentence = get_input()

    if sentence is None:
        return

    tokens = preprocess_text(sentence)

    probability = calculate_sentence_probability(
        tokens,
        conditional_probs
    )

    result_box.configure(
        text=f"""Sentence Probability

Sentence:
{sentence}

Probability:
{probability:.6f}"""
    )


# ----------------------------
# Perplexity
# ----------------------------
def perplex():

    sentence = get_input()

    if sentence is None:
        return

    tokens = preprocess_text(sentence)

    if len(tokens) < 2:
        result_box.configure(
            text=(
                "Perplexity requires at least "
                "one word.\n\n"
                "Example:\n"
                "machine learning"
            )
        )
        return

    value = calculate_perplexity(
        tokens,
        conditional_probs
    )

    result_box.configure(
        text=f"""Perplexity

Sentence:
{sentence}

Perplexity:
{value:.4f}"""
    )


# ----------------------------
# Copy Result
# ----------------------------
def copy_result():

    app.clipboard_clear()

    app.clipboard_append(
        result_box.cget("text")
    )


# ----------------------------
# Clear
# ----------------------------
def clear_all():

    input_box.delete(
        0,
        "end"
    )

    result_box.configure(
        text=(
            "Welcome!\n\n"
            "Choose an action from "
            "the left panel."
        )
    )


# ============================================================
# GUI LAYOUT
# ============================================================

# ----------------------------
# Sidebar
# ----------------------------
sidebar = ctk.CTkFrame(
    app,
    width=260,
    corner_radius=0,
    fg_color="#183153"
)

sidebar.pack(
    side="left",
    fill="y"
)


# ----------------------------
# Main Area
# ----------------------------
main = ctk.CTkFrame(
    app,
    fg_color="#EEF4FB",
    corner_radius=0
)

main.pack(
    side="right",
    fill="both",
    expand=True
)


# ----------------------------
# Sidebar Title
# ----------------------------
ctk.CTkLabel(
    sidebar,
    text="N-Gram\nLanguage Model",
    font=("Helvetica", 26, "bold"),
    text_color="white"
).pack(
    pady=(30, 20)
)


# ----------------------------
# Corpus Selection
# ----------------------------
ctk.CTkLabel(
    sidebar,
    text="Select Corpus",
    text_color="white",
    font=("Helvetica", 13)
).pack()


corpus_menu = ctk.CTkOptionMenu(
    sidebar,
    values=get_corpus_list(),
    command=load_corpus,
    width=210,
    corner_radius=12
)

corpus_menu.pack(
    pady=10
)


if get_corpus_list():
    corpus_menu.set(
        get_corpus_list()[0]
    )


# ----------------------------
# Upload Button
# ----------------------------
ctk.CTkButton(
    sidebar,
    text="Upload Corpus",
    command=upload_corpus,
    width=210,
    height=42,
    corner_radius=12,
    fg_color="#16A34A",
    hover_color="#15803D"
).pack(
    pady=8
)


# ----------------------------
# Input Label
# ----------------------------
ctk.CTkLabel(
    sidebar,
    text="Enter a word or sentence",
    text_color="white",
    font=("Helvetica", 13)
).pack(
    pady=(18, 5)
)


# ----------------------------
# Input Box
# ----------------------------
input_box = ctk.CTkEntry(
    sidebar,
    width=210,
    height=40,
    corner_radius=12,
    placeholder_text="Type here..."
)

input_box.pack(
    pady=10
)


# ----------------------------
# Predict Button
# ----------------------------
ctk.CTkButton(
    sidebar,
    text="Predict Next Word",
    command=predict,
    width=210,
    height=42,
    corner_radius=12
).pack(
    pady=6
)


# ----------------------------
# Probability Button
# ----------------------------
ctk.CTkButton(
    sidebar,
    text="Sentence Probability",
    command=sentence_probability,
    width=210,
    height=42,
    corner_radius=12
).pack(
    pady=6
)


# ----------------------------
# Perplexity Button
# ----------------------------
ctk.CTkButton(
    sidebar,
    text="Perplexity",
    command=perplex,
    width=210,
    height=42,
    corner_radius=12
).pack(
    pady=6
)


# ----------------------------
# Copy Button
# ----------------------------
ctk.CTkButton(
    sidebar,
    text="Copy Result",
    command=copy_result,
    width=210,
    height=42,
    corner_radius=12,
    fg_color="#0EA5E9",
    hover_color="#0284C7"
).pack(
    pady=6
)


# ----------------------------
# Clear Button
# ----------------------------
ctk.CTkButton(
    sidebar,
    text="Clear",
    command=clear_all,
    width=210,
    height=42,
    corner_radius=12,
    fg_color="#64748B",
    hover_color="#475569"
).pack(
    pady=(15, 0)
)


# ============================================================
# MAIN AREA
# ============================================================

# ----------------------------
# Corpus Size
# ----------------------------
corpus_label = ctk.CTkLabel(
    main,
    text="Corpus Size: 0 tokens",
    font=("Helvetica", 14, "bold"),
    text_color="#183153"
)

corpus_label.pack(
    anchor="ne",
    padx=20,
    pady=(20, 5)
)


# ----------------------------
# Vocabulary
# ----------------------------
vocab_label = ctk.CTkLabel(
    main,
    text="Vocabulary: 0 words",
    font=("Helvetica", 13),
    text_color="#183153"
)

vocab_label.pack(
    anchor="ne",
    padx=20
)


# ----------------------------
# Status
# ----------------------------
status = ctk.CTkLabel(
    main,
    text="● Model Ready",
    text_color="#16A34A",
    font=("Helvetica", 13, "bold")
)

status.pack(
    anchor="ne",
    padx=20,
    pady=(0, 10)
)


# ----------------------------
# Result Frame
# ----------------------------
result_frame = ctk.CTkFrame(
    main,
    corner_radius=18,
    fg_color="white"
)

result_frame.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=20
)


# ----------------------------
# Result Box
# ----------------------------
result_box = ctk.CTkLabel(
    result_frame,
    text=(
        "Welcome!\n\n"
        "Choose an action from "
        "the left panel."
    ),
    justify="left",
    anchor="nw",
    font=("Helvetica", 16),
    text_color="#183153"
)

result_box.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=25
)


# ----------------------------
# Footer
# ----------------------------
footer = ctk.CTkLabel(
    main,
    text=(
        "Built with Python • NLTK • "
        "CustomTkinter • N-Gram Language Model"
    ),
    text_color="gray",
    font=("Helvetica", 10)
)

footer.pack(
    pady=(0, 10)
)


# ============================================================
# INITIAL CORPUS
# ============================================================

if get_corpus_list():
    load_corpus(
        get_corpus_list()[0]
    )


# ----------------------------
# Enter = Predict
# ----------------------------
input_box.bind(
    "<Return>",
    lambda e: predict()
)


# ----------------------------
# Start Application
# ----------------------------
app.mainloop()