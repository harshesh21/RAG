# day7_sentence_window.py
import nltk
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Corpus ────────────────────────────────────────────────────────────────────
# Using a longer, denser document so chunking differences actually show up
DOCUMENT = """
Machine learning is a subset of artificial intelligence that enables systems to learn 
from data without being explicitly programmed. It relies on algorithms that iteratively 
learn from data, allowing computers to find hidden insights.

Supervised learning uses labeled training data to learn the mapping between inputs and 
outputs. Common algorithms include linear regression for continuous outputs and logistic 
regression for classification tasks. Support vector machines find the optimal hyperplane 
that separates classes with maximum margin.

Unsupervised learning finds hidden patterns in data without labeled responses. Clustering 
algorithms like k-means group similar data points together. Dimensionality reduction 
techniques like PCA compress data while preserving variance. Autoencoders learn compressed 
representations through neural networks.

Neural networks are computing systems inspired by biological neural networks in animal 
brains. They consist of layers of interconnected nodes that process information. Deep 
learning uses neural networks with many layers to learn hierarchical representations. 
Backpropagation adjusts weights by computing gradients of the loss function.

Transformers revolutionized natural language processing through the attention mechanism. 
Self-attention allows models to weigh the importance of different words in context. 
BERT uses bidirectional attention to understand language context from both directions. 
GPT models use autoregressive generation to predict the next token in a sequence.

Retrieval-augmented generation combines language models with external knowledge retrieval. 
A retriever first finds relevant documents from a large corpus. The generator then 
synthesizes an answer using both the query and retrieved context. RAG reduces hallucination 
by grounding responses in retrieved facts.

Vector databases store and search high-dimensional embeddings efficiently. FAISS uses 
inverted file indexes and product quantization for approximate nearest neighbor search. 
Pinecone and Weaviate offer managed vector search as a service. HNSW graphs provide 
logarithmic search complexity through hierarchical navigation.

Evaluation of RAG systems requires measuring both retrieval and generation quality. 
Precision and recall measure how many relevant documents were retrieved. RAGAS provides 
automated metrics including faithfulness, answer relevancy, and context precision. 
Human evaluation remains the gold standard but does not scale.

Fine-tuning adapts pretrained models to specific domains or tasks. LoRA uses low-rank 
matrix decomposition to reduce the number of trainable parameters. Instruction tuning 
trains models to follow natural language instructions. RLHF aligns model behavior with 
human preferences through reward modeling.

Production RAG systems require monitoring, security, and cost management. Prompt injection 
attacks attempt to override system instructions through user input. Rate limiting and 
caching reduce both latency and cost at scale. Observability tools like LangSmith trace 
each step of the retrieval and generation pipeline.
"""


# ── Part 1A: Sentence-Window Chunker ─────────────────────────────────────────
def build_sentence_windows(text: str, window_size: int = 2) -> list[dict]:
    """
    Returns a list of dicts, each with:
      - 'index_text':    the single sentence to embed
      - 'window_text':   the sentence + surrounding context to send to LLM
      - 'sentence_idx':  position in the document
    """
    sentences = nltk.sent_tokenize(text.strip())
    windows = []

    for i, sentence in enumerate(sentences):
        start = max(0, i - window_size)
        end   = min(len(sentences), i + window_size + 1)
        window = " ".join(sentences[start:end])

        windows.append({
            "index_text":   sentence,
            "window_text":  window,
            "sentence_idx": i,
        })

    return windows


windows = build_sentence_windows(DOCUMENT, window_size=2)

print("=" * 60)
print("PART 1 — Sentence-Window Chunking")
print("=" * 60)
print(f"Total sentences indexed: {len(windows)}")
print()

# Show a sample window so the concept is concrete
sample_idx = 15
s = windows[sample_idx]
print(f"Sample — Sentence #{sample_idx}")
print(f"  INDEX TEXT  (embedded):  {s['index_text']}")
print(f"  WINDOW TEXT (to LLM):    {s['window_text']}")
print()



# ── Part 2: Retrieval Benchmark Harness ──────────────────────────────────────

model = SentenceTransformer("all-MiniLM-L6-v2")

# Test queries — cover different retrieval patterns
# Mix of: exact-match, multi-concept, and edge-case queries
TEST_QUERIES = [
    "How does backpropagation work?",
    "What is the difference between supervised and unsupervised learning?",
    "How do vector databases handle approximate nearest neighbor search?",
    "What metrics are used to evaluate RAG systems?",
    "How does LoRA reduce the number of trainable parameters?",
]

TOP_K = 3  # retrieve top 3 chunks per query


def build_faiss_index(texts: list[str]):
    """Embed texts and return (index, embeddings)."""
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index, embeddings


def retrieve(query: str, index, chunks: list[str], k: int = TOP_K) -> list[str]:
    """Return top-k chunk texts for a query."""
    q_emb = model.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    _, ids = index.search(q_emb.astype(np.float32), k)
    return [chunks[i] for i in ids[0]]


# ── Strategy 1: Fixed chunking (from Day 6) ──────────────────────────────────
splitter_fixed = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
fixed_chunks = splitter_fixed.split_text(DOCUMENT)
fixed_index, _ = build_faiss_index(fixed_chunks)

# ── Strategy 2: Recursive chunking ───────────────────────────────────────────
splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
)
recursive_chunks = splitter_recursive.split_text(DOCUMENT)
recursive_index, _ = build_faiss_index(recursive_chunks)

# ── Strategy 3: Sentence-window — index on sentence, return window ────────────
sw_windows      = build_sentence_windows(DOCUMENT, window_size=2)
sw_index_texts  = [w["index_text"]  for w in sw_windows]
sw_window_texts = [w["window_text"] for w in sw_windows]
sw_index, _     = build_faiss_index(sw_index_texts)

def retrieve_sentence_window(query: str, k: int = TOP_K) -> list[str]:
    """Retrieve by sentence embedding, but return the wider window."""
    q_emb = model.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    _, ids = sw_index.search(q_emb.astype(np.float32), k)
    return [sw_window_texts[i] for i in ids[0]]


# ── Run the benchmark ─────────────────────────────────────────────────────────
print("=" * 60)
print("PART 2 — Chunking Strategy Benchmark")
print("=" * 60)
print(f"Fixed chunks:      {len(fixed_chunks)}")
print(f"Recursive chunks:  {len(recursive_chunks)}")
print(f"Sentence windows:  {len(sw_windows)}")
print()

for query in TEST_QUERIES:
    print("─" * 60)
    print(f"QUERY: {query}")
    print()

    fixed_results     = retrieve(query, fixed_index,     fixed_chunks)
    recursive_results = retrieve(query, recursive_index, recursive_chunks)
    sw_results        = retrieve_sentence_window(query)

    strategies = [
        ("Fixed (200 char)",   fixed_results),
        ("Recursive (400ch)",  recursive_results),
        ("Sentence-Window",    sw_results),
    ]

    for name, results in strategies:
        print(f"  [{name}]")
        for i, chunk in enumerate(results, 1):
            # Truncate for readability; full text is in the variable
            preview = chunk.replace("\n", " ").strip()[:120]
            print(f"    {i}. {preview}...")
        print()


# ── Part 3: Manual Relevance Scoring ─────────────────────────────────────────
# Score each strategy's top result per query: 2=highly relevant, 1=partial, 0=irrelevant
# Fill these in AFTER you run the output and read the results

print("=" * 60)
print("PART 3 — Score Card (fill in after reading output)")
print("=" * 60)
print()
print("For each query, judge the TOP-1 result only.")
print("Scale: 2 = directly answers the query")
print("       1 = related but missing key info")
print("       0 = wrong topic")
print()
print(f"{'Query':<45} {'Fixed':>7} {'Recur':>7} {'SWin':>7}")
print("-" * 70)
for q in TEST_QUERIES:
    label = q[:44]
    print(f"  {label:<44}  _____  _____  _____")
print()
print("After scoring, tally each column.")
print("The strategy with the highest total wins on this corpus.")
print("Note: window SIZE matters too — re-run with window_size=1 and window_size=3")