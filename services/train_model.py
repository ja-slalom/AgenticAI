import spacy
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Sample dataset
"""
Token Count (num_tokens): Measures text length.
Sentence Count (num_sentences): Identifies sentence complexity.
Unique Part-of-Speech (POS) Tags (unique_pos_tags): Captures linguistic variety.
"""
data = [
    ("What is AI?", 3, 1, 2, 5),
    ("Define AI.", 2, 1, 2, 3),
    ("What is data?", 3, 1, 2, 4),
    ("Name a use of AI.", 4, 1, 2, 5),
    ("Explain IoT.", 3, 1, 2, 4),
    ("List AI types.", 3, 1, 2, 4),
    ("What is NLP?", 3, 1, 2, 4),
    ("Describe ML.", 3, 1, 2, 4),
    ("Give an example of AI.", 5, 1, 2, 6),
    ("Name a machine learning algorithm.", 5, 1, 3, 6),
    ("What is automation?", 3, 1, 2, 4),
    ("Explain how AI differs from machine learning.", 9, 1, 5, 35),
    ("Discuss the ethical implications of artificial intelligence in modern society.", 12, 1, 7, 55),
    ("Compare and contrast various optimization techniques in deep learning architectures.", 15, 1, 8, 80),
    ("Define natural language processing and its applications.", 8, 1, 4, 30),
    ("What are the key differences between supervised and unsupervised learning?", 11, 1, 6, 50),
    ("Illustrate how transformers improve upon traditional RNN models in NLP tasks.", 13, 1, 7, 65),
    ("Describe the role of reinforcement learning in developing autonomous agents.", 12, 1, 6, 60),
    ("Summarize recent advancements in multimodal AI systems combining text and image understanding.", 16, 1, 9, 85),
    ("Analyze the social impact of large language models on information dissemination and bias.", 14, 1, 8, 75),
    ("Explore the limitations of explainable AI in high-stakes decision-making environments.", 15, 1, 8, 78),
    ("What challenges arise when deploying machine learning models in production?", 10, 1, 5, 40),
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["prompt", "token_count", "sentence_count", "unique_pos_tags", "complexity_score"])

# Define features (X) and target variable (y)
X = df[["token_count", "sentence_count", "unique_pos_tags"]]
y = df["complexity_score"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

# Save the trained model
joblib.dump(model, "../model/complexity_model.pkl")
print("Model saved successfully!")