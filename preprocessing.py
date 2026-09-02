import collections
import re

class SmartNgramPredictor:
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = collections.defaultdict(collections.Counter)
        
    def train(self, text_corpus):
        text = text_corpus.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        
        for i in range(len(words) - self.n + 1):
            context = tuple(words[i : i + self.n - 1])
            next_word = words[i + self.n - 1]
            self.ngram_counts[context][next_word] += 1

    def predict_next_word(self, input_text, top_k=3):
        words = input_text.lower().split()
        context = tuple(words[-(self.n - 1):])
        
        predictions = self.ngram_counts.get(context, None)
        
        if not predictions:
            return ["No prediction available"]
        
        return [word for word, count in predictions.most_common(top_k)]

# ===== Data =====
sample_corpus = """
natural language processing is fun. natural language processing helps computers understand human language.
next word prediction uses n-gram models. next word prediction with python is powerful.
"""


model = SmartNgramPredictor(n=3)
model.train(sample_corpus)


input_phrase = "next word"
results = model.predict_next_word(input_phrase, top_k=2)

print(f"Input: '{input_phrase}'")
print("Top Predicted Next Words:", results)

print("\n=== Next Word Predictor ready! (ထွက်ချင်ရင် 'exit' ဟု ရိုက်ပါ) ===")
while True:
    user_input = input("\nInput စာလုံး ၂ လုံး ရိုက်ပါ: ")
    if user_input.lower() == 'exit':
        break
    
    predictions = model.predict_next_word(user_input, top_k=3)
    print("Next predicted words:", predictions)