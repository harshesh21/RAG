# day8_document_parsing.py
import json
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title, NarrativeText, Table, ListItem, Header, Footer, Text
)

PDF_PATH = "document.pdf"

# ── Part 1A: See what raw extraction gives you ────────────────────────────────
import pdfminer.high_level as pdfminer

raw_text = pdfminer.extract_text(PDF_PATH)

print("=" * 60)
print("PART 1A — Raw pdfminer extraction (first 800 chars)")
print("=" * 60)
print(raw_text[:800])
print()

# ── Part 1B: Unstructured element detection ───────────────────────────────────
elements = partition_pdf(PDF_PATH)

print("=" * 60)
print("PART 1B — Unstructured element types found")
print("=" * 60)

from collections import Counter
type_counts = Counter(type(el).__name__ for el in elements)
for el_type, count in type_counts.most_common():
    print(f"  {el_type:<20} {count}")
print(f"\n  Total elements: {len(elements)}")
print()

# Show first 10 elements with their types
print("First 10 elements:")
for el in elements[:10]:
    preview = str(el)[:80].replace("\n", " ")
    print(f"  [{type(el).__name__:<15}] {preview}")


# ── Part 2: Filter noise elements ─────────────────────────────────────────────

KEEP_TYPES = (Title, NarrativeText, Table, ListItem, Text)
DROP_TYPES = (Header, Footer)

def filter_elements(elements):
    """Drop headers, footers, and very short noise elements."""
    filtered = []
    for el in elements:
        if isinstance(el, DROP_TYPES):
            continue
        text = str(el).strip()
        if len(text) < 20:          # drop page numbers, stray characters
            continue
        filtered.append(el)
    return filtered


filtered = filter_elements(elements)

print("=" * 60)
print("PART 2 — After filtering")
print("=" * 60)
print(f"  Before: {len(elements)} elements")
print(f"  After:  {len(filtered)} elements")
print(f"  Dropped: {len(elements) - len(filtered)}")
print()

# Show element type breakdown after filtering
type_counts_filtered = Counter(type(el).__name__ for el in filtered)
for el_type, count in type_counts_filtered.most_common():
    print(f"  {el_type:<20} {count}")




# ── Part 3: Chunk with metadata ───────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

def build_chunks_with_metadata(elements, source_filename: str) -> list[dict]:
    """
    Returns list of dicts:
      - text:           chunk text to embed
      - metadata:       source, page, section_title, element_type
    """
    chunks = []
    current_section = "Introduction"   # default if no title seen yet

    for el in elements:
        # Track section title as we walk through elements
        if isinstance(el, Title) and is_valid_title(str(el).strip()):
            current_section = str(el).strip()

        text = str(el).strip()
        if not text:
            continue

        # Get page number from unstructured metadata if available
        page_num = None
        if hasattr(el, "metadata") and hasattr(el.metadata, "page_number"):
            page_num = el.metadata.page_number

        # Split element text into chunks
        cleaned = clean_text(text)
        sub_chunks = splitter.split_text(cleaned)

        for chunk_text in sub_chunks:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source":        source_filename,
                    "page":          page_num,
                    "section_title": current_section,
                    "element_type":  type(el).__name__,
                }
            })

    return chunks

import unicodedata

def clean_text(text: str) -> str:
    # Normalize unicode ligatures (ﬁ→fi, ﬂ→fl, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text

def is_valid_title(text: str) -> bool:
    # Reject titles that are too long (likely a sentence, not a heading)
    if len(text) > 60:
        return False
    # Reject titles that contain commas (likely a list or sentence fragment)
    if text.count(",") >= 2:
        return False
    return True

chunks = build_chunks_with_metadata(filtered, PDF_PATH)

print("=" * 60)
print("PART 3 — Chunks with metadata")
print("=" * 60)
print(f"Total chunks: {len(chunks)}")
print()

# Show 3 sample chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"  Chunk {i+1}:")
    print(f"    text:     {chunk['text'][:100].replace(chr(10), ' ')}...")
    print(f"    metadata: {json.dumps(chunk['metadata'])}")
    print()

# ── Part 4: Build index and retrieve with metadata ────────────────────────────
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

def retrieve_with_metadata(query: str, k: int = 3) -> list[dict]:
    q_emb = model.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    scores, ids = index.search(q_emb.astype(np.float32), k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({
            "score":    round(float(score), 4),
            "text":     chunks[idx]["text"],
            "metadata": chunks[idx]["metadata"],
        })
    return results


# Test queries — adjust to match your actual PDF content
TEST_QUERIES = [
    "What is RAG and how does it work?",
    "What AWS services are recommended for RAG?",
    "What are the benefits and drawbacks of RAG architectures?",
]

print("=" * 60)
print("PART 4 — Retrieval with metadata")
print("=" * 60)

for query in TEST_QUERIES:
    print(f"\nQUERY: {query}")
    results = retrieve_with_metadata(query)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [score={r['score']}] "
              f"[page={r['metadata']['page']}] "
              f"[section={r['metadata']['section_title'][:40]}]")
        print(f"     {r['text'][:100].replace(chr(10), ' ')}...")