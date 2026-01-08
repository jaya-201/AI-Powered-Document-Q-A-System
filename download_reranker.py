"""
Script to download and cache the cross-encoder model
Run this once to download the model to local cache
"""
from sentence_transformers import CrossEncoder

print("Downloading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
print("This may take a few minutes...")

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("\n✓ Model downloaded and cached successfully!")
print("The model is now available for use in the RAG pipeline.")

# Test the model
test_pairs = [
    ["What is machine learning?", "Machine learning is a branch of artificial intelligence"],
    ["What is machine learning?", "The weather is sunny today"]
]

scores = model.predict(test_pairs)
print(f"\nTest scores: {scores}")
print("Higher score = more relevant match")
