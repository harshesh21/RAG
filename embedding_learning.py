from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat.",
    "A dog rested on the rug.",
    "Python is a programming language.",
    "JavaScript runs in the browser.",
    "Machine learning uses data to make predictions.",
    "Deep learning is a subset of machine learning.",
    "The weather is sunny today.",
    "It is a bright and clear day outside.",
]

embeddings = model.encode(sentences)

print(f"Embedding shape: {embeddings.shape}")
print(f"Printing for first embedding {embeddings[0].size}")
print(f"Each sentence = vector of {embeddings.shape[1]} numbers\n")

from sklearn.metrics.pairwise import cosine_similarity

# Compare every sentence to every other sentence
similarity_matrix = cosine_similarity(embeddings)

print("Similarity scores (1.0 = identical meaning, 0.0 = unrelated):\n")
for i, sent_a in enumerate(sentences):
    for j, sent_b in enumerate(sentences):
        if j <= i:
            continue
        score = similarity_matrix[i][j]
        if score > 0.5:  # only show high similarity pairs
            print(f"  {score:.3f}  |  '{sent_a[:40]}' ↔ '{sent_b[:40]}'")

import umap
import matplotlib.pyplot as plt

# Reduce 384 dimensions down to 2 so we can plot it
reducer = umap.UMAP(n_neighbors=5, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Color code by topic
colors = ['#E85D24','#E85D24',      # cat/dog = orange
          '#185FA5','#185FA5',      # Python/JS = blue  
          '#0F6E56','#0F6E56',      # ML sentences = green
          '#993556','#993556']      # weather = pink

plt.figure(figsize=(10, 7))
for i, (x, y) in enumerate(embeddings_2d):
    plt.scatter(x, y, color=colors[i], s=100, zorder=2)
    plt.annotate(
        sentences[i][:30] + "...",
        (x, y),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=8
    )

plt.title("Sentence embeddings visualized in 2D (UMAP)")
plt.axis('off')
plt.tight_layout()
plt.show()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product_sim(a, b):
    return np.dot(a, b)

query = model.encode(["animal sitting on something"])
doc1 = embeddings[0]  # "The cat sat on the mat"
doc2 = embeddings[2]  # "Python is a programming language"

print("Query: 'animal sitting on something'\n")
print(f"  Cosine vs cat sentence:    {cosine_sim(query[0], doc1):.4f}")
print(f"  Cosine vs Python sentence: {cosine_sim(query[0], doc2):.4f}")
print()
print(f"  Dot product vs cat sentence:    {dot_product_sim(query[0], doc1):.4f}")
print(f"  Dot product vs Python sentence: {dot_product_sim(query[0], doc2):.4f}")


