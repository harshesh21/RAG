"""
Day 9 — First Full End-to-End RAG Pipeline
Goal: retrieve top-K chunks → build prompt → call Claude 3 Haiku on Bedrock → return answer with citations

Builds directly on Day 8's parsing + FAISS index with metadata.
Run sections top-to-bottom, or use rag_answer() as a callable.
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import json
import re
import textwrap
from pathlib import Path

import boto3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from unstructured.partition.pdf import partition_pdf   # your Day 8 parser
# If you saved your index/chunks from Day 8, skip Section 1 and load them.

from build_index_pypdf import build_index_from_pdf   # your Day 8 parser (pypdf version)
# ── 1. Rebuild Index from Day 8 (skip if you serialized the index) ────────────
# Paste / import your Day 8 parsing + chunking function here.
# Minimal version shown — swap in your actual pipeline.

def build_index_from_pdf_linux(pdf_path: str, chunk_size: int = 1024, overlap: int = 128):
    """
    Re-runs your Day 8 pipeline and returns (chunks, metadata, faiss_index, bm25).
    chunks    — list of str
    metadata  — list of dict  {source, page, section_title}
    index     — faiss.IndexFlatIP (inner product on L2-normed vectors = cosine)
    bm25      — BM25Okapi for hybrid search
    """
    from unstructured.partition.pdf import partition_pdf
    from unstructured.cleaners.core import clean_extra_whitespace
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import unicodedata

    print(f"[index] Parsing {pdf_path} ...")
    elements = partition_pdf(pdf_path, strategy="auto")


    # Filter + normalize (your Day 8 logic)
    keep_types = {"Title", "NarrativeText", "ListItem", "Table"}
    cleaned = []
    current_title = "Unknown"
    pending_title = None          # ← initialize before loop

    for el in elements:
        category = el.category if hasattr(el, "category") else type(el).__name__

        if category not in keep_types:
            continue

        text = clean_extra_whitespace(str(el))   # ← always get current element text first

        if category == "Title":
            current_title = text
            pending_title = text
            continue                              # ← don't index bare title elements

        if category == "NarrativeText" and pending_title:
            text = clean_extra_whitespace(pending_title + ". " + text)
            pending_title = None

        text = unicodedata.normalize("NFKC", text)
        if len(text) < 20:
            continue

        page = el.metadata.page_number if el.metadata and el.metadata.page_number else 0
        cleaned.append({"text": text, "page": page, "section_title": current_title})
    # Chunk with metadata inheritance
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks, metadata = [], []
    for item in cleaned:
        for chunk in splitter.split_text(item["text"]):
            # In your parsing pipeline, after chunking:
            if len(chunk.split()) >= 15:
                chunks.append(chunk)
                metadata.append({
                    "source": Path(pdf_path).name,
                    "page": item["page"],
                    "section_title": item["section_title"],
                })

    # Embed + FAISS (cosine via inner product on normed vectors)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[index] Embedding {len(chunks)} chunks ...")
    vecs = model.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    vecs = np.array(vecs, dtype="float32")
    print(f"[diag] len(cleaned): {len(cleaned)}")
    print(f"[diag] len(chunks): {len(chunks)}")
    print(f"[diag] vecs shape: {vecs.shape}")
    if len(chunks) == 0:
        raise ValueError("No chunks survived. Check parsing and filter steps above.")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # BM25 for hybrid
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    print(f"[index] Done. {len(chunks)} chunks indexed.")
    return chunks, metadata, index, bm25, model


# ── 2. Retrieval — Hybrid search (your Day 5 alpha-weighted approach) ─────────

def retrieve(
    query: str,
    chunks: list[str],
    metadata: list[dict],
    faiss_index,
    bm25,
    embed_model,
    k: int = 5,
    alpha: float = 0.5,   # 0 = pure BM25, 1 = pure dense
) -> list[dict]:
    """
    Returns top-k results as list of dicts:
        {rank, text, score, source, page, section_title}
    """
    n = len(chunks)

    # Dense scores (cosine, already normed)
    q_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    dense_scores_raw, dense_ids = faiss_index.search(q_vec, n)
    dense_scores = np.zeros(n)
    for score, idx in zip(dense_scores_raw[0], dense_ids[0]):
        if idx != -1:
            dense_scores[idx] = score

    # BM25 scores
    bm25_scores = np.array(bm25.get_scores(query.lower().split()), dtype="float32")

    # Normalize both to [0, 1]
    def minmax(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    combined = alpha * minmax(dense_scores) + (1 - alpha) * minmax(bm25_scores)
    top_ids = combined.argsort()[::-1][:k]

    return [
        {
            "rank": i + 1,
            "text": chunks[idx],
            "score": float(combined[idx]),
            "source": metadata[idx]["source"],
            "page": metadata[idx]["page"],
            "section_title": metadata[idx]["section_title"],
        }
        for i, idx in enumerate(top_ids)
    ]


# ── 3. Prompt Builder ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise technical assistant. Answer questions using ONLY the context blocks provided.

Rules:
- Cite every factual claim with [Source: <filename>, p.<page>].
- If the context does not contain enough information, say exactly: "The provided documents do not contain sufficient information to answer this question."
- Do not speculate or add information from outside the context.
- Be concise. Prefer bullet points for multi-part answers."""


def build_prompt(query: str, results: list[dict]) -> str:
    """
    Assembles the user turn: numbered context blocks + the question.
    Each block carries its citation label so the model can reference it inline.
    """
    context_blocks = []
    for r in results:
        block = (
            f"[CONTEXT {r['rank']}] "
            f"Source: {r['source']}, p.{r['page']}, Section: \"{r['section_title']}\"\n"
            f"{r['text']}"
        )
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)

    return (
        f"Use the following context to answer the question.\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite sources inline):"
    )


# ── 4. Bedrock Call ───────────────────────────────────────────────────────────

def call_bedrock(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
    max_tokens: int = 512,
    temperature: float = 0.0,    # deterministic for RAG; crank up for summaries
) -> str:
    """
    Calls Claude 3 Haiku on Bedrock and returns the answer string.
    Uses the Messages API format (invoke_model with anthropic bedrock body).
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


# ── 5. Orchestrator ───────────────────────────────────────────────────────────

def rag_answer(
    query: str,
    chunks: list[str],
    metadata: list[dict],
    faiss_index,
    bm25,
    embed_model,
    k: int = 5,
    alpha: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Full RAG pipeline: query → retrieve → prompt → Bedrock → cited answer.

    Returns:
        {
            "query":    str,
            "answer":   str,
            "sources":  list[dict],   # the top-k retrieval results
            "prompt":   str,          # the exact prompt sent to the model
        }
    """
    # Step 1: Retrieve
    results = retrieve(query, chunks, metadata, faiss_index, bm25, embed_model, k=k, alpha=alpha)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"\nTop-{k} retrieved chunks:")
        for r in results:
            print(f"  [{r['rank']}] score={r['score']:.3f} | p.{r['page']} | {r['section_title'][:50]}")
            print(f"       {r['text'][:120].replace(chr(10), ' ')} ...")

    # Step 2: Build prompt
    prompt = build_prompt(query, results)

    if verbose:
        print(f"\n{'─'*60}")
        print("Calling Claude 3 Haiku on Bedrock ...")

    # Step 3: Call model
    answer = call_bedrock(prompt)

    if verbose:
        print(f"\n{'─'*60}")
        print("ANSWER:\n")
        print(textwrap.fill(answer, width=80))
        print(f"\nSOURCES:")
        seen = set()
        for r in results:
            key = (r["source"], r["page"])
            if key not in seen:
                print(f"  • {r['source']} — p.{r['page']} — {r['section_title']}")
                seen.add(key)

    return {"query": query, "answer": answer, "sources": results, "prompt": prompt}


# ── 6. Run the Lab ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 6a. Build (or load) the index ────────────────────────────────────────
    PDF_PATH = "ragAws.pdf"    # ← your Day 8 PDF

    # Option A: rebuild fresh (slow, ~30s)
    chunks, metadata, faiss_index, bm25, embed_model = build_index_from_pdf(PDF_PATH)

    # Option B: load a saved index (fast — add serialization if you want it)
    # chunks, metadata, faiss_index, bm25, embed_model = load_index("day8_index/")

    # ── 6b. Test queries — vary these to probe retrieval quality ─────────────
    test_queries = [
        # Fact lookup — should retrieve a tight passage
        "What embedding model does AWS recommend for RAG on Bedrock?",

        # Multi-hop — requires synthesizing across chunks
        "What are the main components of a RAG architecture on AWS?",

        # Edge case — out-of-scope; model should refuse rather than hallucinate
        "What is the capital of France?",
    ]

    results_log = []
    for query in test_queries:
        result = rag_answer(
            query,
            chunks, metadata, faiss_index, bm25, embed_model,
            k=5,
            alpha=0.5,
            verbose=True,
        )
        results_log.append(result)
        print("\n")

    # ── 6c. Spot-check: inspect the exact prompt for one query ────────────────
    print("\n" + "="*60)
    print("EXACT PROMPT SENT FOR QUERY 1 (for debugging):\n")
    print(results_log[1]["prompt"])


# ── 7. What to Observe ────────────────────────────────────────────────────────
"""
Things to check and note for your log:

RETRIEVAL QUALITY
- Do the top-5 chunks actually contain the answer?
- Do retrieval scores drop sharply after rank 2–3? (Good sign — tight retrieval)
- Does the section_title help you quickly spot if a chunk is from the right part of the doc?

ANSWER QUALITY
- Does the model cite [Source: ..., p.X] inline?
- Does it stay within the context, or does it hallucinate details not in the chunks?
- On the out-of-scope question ("capital of France"), does it refuse correctly?
# 
PROMPT DESIGN
- Try changing k from 5 → 3. Does answer quality drop or improve?
  Fewer chunks = less noise but higher miss rate.
- Try alpha=0.0 (pure BM25) vs alpha=1.0 (pure dense). Compare answer sources.
- Try temperature=0.3 vs 0.0. Does the phrasing change meaningfully?

KNOWN FAILURE MODES TO HIT INTENTIONALLY
- Ask something vague: "Tell me about RAG."
  → Model may over-summarize or cherry-pick. Note what chunks it pulls.
- Ask something the whitepaper covers obliquely: "What are the cost tradeoffs?"
  → Tests whether the model extrapolates beyond evidence or holds the line.

LOG THESE IN YOUR NOTES FOR DAY 10:
  best_k, best_alpha, any prompt tweaks that improved answer quality.
  Day 10 will make these configurable and add no-answer handling.
"""
