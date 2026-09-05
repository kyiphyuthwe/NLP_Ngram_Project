from src.preprocessing import preprocess_text


with open("data/corpus.txt", "r") as file:
    text = file.read()

tokens = preprocess_text(text)

print("Number of words:", len(tokens))
print("First 30 words:")
print(tokens[:30])