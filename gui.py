import customtkinter as ctk
from tkinter import filedialog

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
app.geometry("980x620")
app.resizable(False, False)

# ----------------------------
# Load Initial Corpus
# ----------------------------
with open("data/corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()

corpus_tokens = preprocess_text(text)
bigrams = generate_ngrams(corpus_tokens, 2)
conditional_probs = calculate_conditional_probabilities(bigrams)

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


def upload_corpus():
    global corpus_tokens, bigrams, conditional_probs

    path = filedialog.askopenfilename(
        title="Select Corpus",
        filetypes=[("Text Files", "*.txt")]
    )

    if not path:
        return

    with open(path, "r", encoding="utf-8") as f:
        new_text = f.read()

    corpus_tokens = preprocess_text(new_text)
    bigrams = generate_ngrams(corpus_tokens, 2)
    conditional_probs = calculate_conditional_probabilities(bigrams)

    corpus_label.configure(
        text=f"Corpus Size: {len(corpus_tokens)} words"
    )

    result_box.configure(
        text=f"New corpus loaded successfully!\n\nTotal Words: {len(corpus_tokens)}"
    )


def predict():
    word = get_input()

    if word is None:
        return

    predicted, probability = predict_next_word(
        (word.lower(),),
        conditional_probs
    )

    if predicted is None:
        result_box.configure(
            text=f"Prediction\n\nInput: {word}\n\nNo prediction found."
        )
        return

    result_box.configure(
        text=f"""Prediction

Input:
{word}

Next Word:
{predicted}

Confidence:
{probability:.4f}"""
    )


def sentence_probability():
    sentence = get_input()

    if sentence is None:
        return

    sentence_tokens = preprocess_text(sentence)

    probability = calculate_sentence_probability(
        sentence_tokens,
        conditional_probs
    )

    result_box.configure(
        text=f"""Sentence Probability

Sentence:
{sentence}

Probability:
{probability:.6f}"""
    )


def perplexity():
    sentence = get_input()

    if sentence is None:
        return

    sentence_tokens = preprocess_text(sentence)

    value = calculate_perplexity(
        sentence_tokens,
        conditional_probs
    )

    result_box.configure(
        text=f"""Perplexity

Sentence:
{sentence}

Perplexity:
{value:.4f}"""
    )


def copy_result():
    app.clipboard_clear()
    app.clipboard_append(result_box.cget("text"))


def clear_all():
    input_box.delete(0, "end")

    result_box.configure(
        text="Welcome!\n\nChoose an action from the left panel."
    )

# ----------------------------
# Layout
# ----------------------------

sidebar = ctk.CTkFrame(
    app,
    width=260,
    corner_radius=0,
    fg_color="#183153"
)

sidebar.pack(side="left", fill="y")

main = ctk.CTkFrame(
    app,
    fg_color="#EEF4FB",
    corner_radius=0
)

main.pack(side="right", fill="both", expand=True)

# ----------------------------
# Sidebar
# ----------------------------

title = ctk.CTkLabel(
    sidebar,
    text="N-Gram\nLanguage Model",
    font=("Helvetica", 26, "bold"),
    text_color="white"
)

title.pack(pady=(30, 15))

ctk.CTkLabel(
    sidebar,
    text="Enter a word or sentence",
    text_color="white",
    font=("Helvetica", 13)
).pack()

input_box = ctk.CTkEntry(
    sidebar,
    width=210,
    height=40,
    corner_radius=12,
    placeholder_text="Type here..."
)

input_box.pack(pady=15)

button_width = 210
button_height = 42

ctk.CTkButton(
    sidebar,
    text="Upload Corpus",
    command=upload_corpus,
    width=button_width,
    height=button_height,
    corner_radius=12,
    fg_color="#16A34A",
    hover_color="#15803D"
).pack(pady=6)

ctk.CTkButton(
    sidebar,
    text="Predict Next Word",
    command=predict,
    width=button_width,
    height=button_height,
    corner_radius=12
).pack(pady=6)

ctk.CTkButton(
    sidebar,
    text="Sentence Probability",
    command=sentence_probability,
    width=button_width,
    height=button_height,
    corner_radius=12
).pack(pady=6)

ctk.CTkButton(
    sidebar,
    text="Perplexity",
    command=perplexity,
    width=button_width,
    height=button_height,
    corner_radius=12
).pack(pady=6)

ctk.CTkButton(
    sidebar,
    text="Copy Result",
    command=copy_result,
    width=button_width,
    height=button_height,
    corner_radius=12,
    fg_color="#0EA5E9",
    hover_color="#0284C7"
).pack(pady=6)

ctk.CTkButton(
    sidebar,
    text="Clear",
    command=clear_all,
    width=button_width,
    height=button_height,
    corner_radius=12,
    fg_color="#64748B",
    hover_color="#475569"
).pack(pady=(15, 0))

# ----------------------------
# Main Content
# ----------------------------

corpus_label = ctk.CTkLabel(
    main,
    text=f"Corpus Size: {len(corpus_tokens)} words",
    font=("Helvetica", 14, "bold"),
    text_color="#183153"
)

corpus_label.pack(anchor="ne", padx=20, pady=(20, 5))

status = ctk.CTkLabel(
    main,
    text="● Model Ready",
    text_color="#16A34A",
    font=("Helvetica", 13, "bold")
)

status.pack(anchor="ne", padx=20)

result_frame = ctk.CTkFrame(
    main,
    corner_radius=18,
    fg_color="white"
)

result_frame.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=25
)

result_box = ctk.CTkLabel(
    result_frame,
    text="Welcome!\n\nChoose an action from the left panel.",
    justify="left",
    anchor="nw",
    font=("Helvetica", 16),
    text_color="#183153"
)

result_box.pack(fill="both", expand=True, padx=25, pady=25)

footer = ctk.CTkLabel(
    main,
    text="Built with Python • CustomTkinter • N-Gram Language Model",
    text_color="gray",
    font=("Helvetica", 10)
)

footer.pack(pady=(0, 10))

input_box.bind("<Return>", lambda e: predict())

app.mainloop()