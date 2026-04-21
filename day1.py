import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Simple sentence
tokens = ["the", "cat", "sat", "on", "the", "mat"]

# Random Q, K, V vectors (normally learned during training)
np.random.seed(42)
d = 8  # small embedding size for demo
Q = np.random.randn(len(tokens), d)
K = np.random.randn(len(tokens), d)
V = np.random.randn(len(tokens), d)

# Compute attention scores for token "cat" (index 1)
query = Q[1]
scores = [np.dot(query, K[i]) / np.sqrt(d) for i in range(len(tokens))]
weights = softmax(scores)

print("Attention weights from 'cat' to each token:\n")
for token, weight in zip(tokens, weights):
    bar = "█" * int(weight * 40)
    print(f"  {token:6s}  {weight:.3f}  {bar}")