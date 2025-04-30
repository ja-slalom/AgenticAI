import os
import spacy
import joblib

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "complexity_model.pkl")
model = joblib.load(model_path)

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Function to predict complexity dynamically
def predict_complexity(prompt: str) -> float:
    doc = nlp(prompt)
    num_tokens = len(doc)
    num_sentences = len(list(doc.sents))
    unique_pos_tags = len(set(token.pos_ for token in doc))

    return model.predict([[num_tokens, num_sentences, unique_pos_tags]])[0]


# Select model dynamically
def select_model(prompt: str) -> str:
    complexity = predict_complexity(prompt)
    print(f"Predicted complexity score: {complexity}")
    if complexity < 20:
        return "o1-mini"
    elif complexity < 60:
        return "gpt-4"
    else:
        return "gpt-4o"


if __name__ == "__main__":
    test_prompt = "What is the job?"
    selected_model = select_model(test_prompt)
    print(f"Selected model for prompt: {selected_model}")