from rank_bm25 import BM25Okapi

# Simulate a product catalog
docs = [
    "SKU-9042 wireless bluetooth headphones noise cancelling",
    "SKU-1138 running shoes lightweight breathable mesh",
    "SKU-7734 laptop stand adjustable aluminum ergonomic",
    "SKU-3301 coffee maker programmable 12 cup stainless",
    "SKU-9042 replacement ear cushions compatible headphones",
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks with many layers",
    "natural language processing helps computers understand text",
]

# Tokenize — BM25 works on raw words not vectors
tokenized = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized)

# Query 1 — exact product code
query_exact = "SKU-9042"
scores = bm25.get_scores(query_exact.split())
print("Query: 'SKU-9042' (exact product code)\n")
for doc, score in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:4]:
    print(f"  {score:.3f}  {doc[:60]}")


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

def dense_search(query, top_k=4):
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]

def bm25_search(query, top_k=4):
    scores = bm25.get_scores(query.split())
    results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Test 1 — exact product code
print("=" * 60)
print("Query: 'SKU-9042'\n")
print("BM25 results:")
for doc, score in bm25_search("SKU-9042"):
    print(f"  {score:.3f}  {doc[:60]}")

print("\nDense results:")
for doc, score in dense_search("SKU-9042"):
    print(f"  {score:.3f}  {doc[:60]}")

# Test 2 — semantic query
print("\n" + "=" * 60)
print("Query: 'headphones for blocking out noise'\n")
print("BM25 results:")
for doc, score in bm25_search("headphones for blocking out noise"):
    print(f"  {score:.3f}  {doc[:60]}")

print("\nDense results:")
for doc, score in dense_search("headphones for blocking out noise"):
    print(f"  {score:.3f}  {doc[:60]}")

# Test 3 — AI concept query  
print("\n" + "=" * 60)
print("Query: 'AI and neural networks'\n")
print("BM25 results:")
for doc, score in bm25_search("AI and neural networks"):
    print(f"  {score:.3f}  {doc[:60]}")

print("\nDense results:")
for doc, score in dense_search("AI and neural networks"):
    print(f"  {score:.3f}  {doc[:60]}")

def hybrid_search(query, top_k=4, alpha=0.5):
    # alpha = 0.0 means pure BM25, 1.0 means pure dense
    
    # BM25 scores (normalize to 0-1)
    bm25_scores = bm25.get_scores(query.split())
    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = bm25_scores / bm25_max

    # Dense scores
    query_emb = model.encode([query])
    dense_scores = cosine_similarity(query_emb, embeddings)[0]

    # Combine
    combined = (1 - alpha) * bm25_norm + alpha * dense_scores
    results = sorted(zip(docs, combined), key=lambda x: x[1], reverse=True)
    return results[:top_k]


print("Hybrid search — 'SKU-9042'\n")
for doc, score in hybrid_search("SKU-9042", alpha=0.3):
    print(f"  {score:.3f}  {doc[:60]}")

print("\nHybrid search — 'headphones for blocking out noise'\n")
for doc, score in hybrid_search("headphones for blocking out noise", alpha=0.7):
    print(f"  {score:.3f}  {doc[:60]}")

print("\nHybrid search — 'AI and neural networks' (alpha=0.7, favors dense)\n")
for doc, score in hybrid_search("AI and neural networks", alpha=0.7):
    print(f"  {score:.3f}  {doc[:60]}")