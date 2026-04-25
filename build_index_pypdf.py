"""
build_index_from_pdf — pypdf version
Drop-in replacement for the Unstructured version.
No tesseract, no poppler, no system dependencies.
"""

import unicodedata
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def looks_like_title(line: str) -> bool:
    """
    Simple heuristic: a line is probably a section title if it is:
    - Short (under 80 chars)
    - Does not end with a period (titles rarely do)
    - Not all uppercase (that's usually a header/footer)
    - Has at least 3 words
    """
    line = line.strip()
    if not line:
        return False
    if len(line) > 80:
        return False
    if line.endswith("."):
        return False
    words = line.split()
    if len(words) < 3:
        return False
    return True


def build_index_from_pdf(pdf_path: str, chunk_size: int = 1024, overlap: int = 256):
    """
    Parses a PDF with pypdf, chunks the text, embeds with sentence-transformers,
    and returns everything needed for hybrid search.

    Returns:
        chunks      — list of str
        metadata    — list of dict {source, page, section_title}
        faiss_index — faiss.IndexFlatIP (cosine via normed inner product)
        bm25        — BM25Okapi
        model       — SentenceTransformer (reuse for query encoding)
    """
    reader = PdfReader(pdf_path)
    print(f"[index] Parsing {pdf_path} — {len(reader.pages)} pages ...")

    cleaned = []
    current_title = "Introduction"

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text()
        if not raw:
            continue

        # Normalize unicode (fixes ligatures like ﬃ → ffi)
        raw = unicodedata.normalize("NFKC", raw)

        # Split into lines to detect section titles
        lines = raw.split("\n")
        page_text_parts = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip running header (repeated on every page in this PDF)
            if "AWS Prescriptive Guidance" in line and len(line) < 100:
                continue
            # Skip page numbers (lone digits)
            if line.isdigit():
                continue

            if looks_like_title(line):
                # Flush accumulated text before starting new section
                if page_text_parts:
                    text = " ".join(page_text_parts).strip()
                    if len(text.split()) >= 15:
                        cleaned.append({
                            "text": text,
                            "page": page_num,
                            "section_title": current_title,
                        })
                    page_text_parts = []
                current_title = line
            else:
                page_text_parts.append(line)

        # Flush remaining text at end of page
        if page_text_parts:
            text = " ".join(page_text_parts).strip()
            if len(text.split()) >= 15:
                cleaned.append({
                    "text": text,
                    "page": page_num,
                    "section_title": current_title,
                })

    print(f"[index] {len(cleaned)} text blocks after parsing.")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks, metadata = [], []

    for item in cleaned:
        for chunk in splitter.split_text(item["text"]):
            if len(chunk.split()) >= 15:   # drop orphaned fragments
                chunks.append(chunk)
                metadata.append({
                    "source": Path(pdf_path).name,
                    "page": item["page"],
                    "section_title": item["section_title"],
                })

    print(f"[index] {len(chunks)} chunks after splitting + filter.")

    if len(chunks) == 0:
        raise ValueError("No chunks survived. Check parsing above.")

    # ── Embed ─────────────────────────────────────────────────────────────────
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[index] Embedding {len(chunks)} chunks ...")
    vecs = model.encode(
        chunks,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vecs = np.array(vecs, dtype="float32")

    # ── FAISS (cosine via inner product on L2-normed vectors) ─────────────────
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # ── BM25 ──────────────────────────────────────────────────────────────────
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    print(f"[index] Done. {len(chunks)} chunks indexed.")
    return chunks, metadata, index, bm25, model


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunks, metadata, faiss_index, bm25, embed_model = build_index_from_pdf("document.pdf")

    # Spot-check: print first 5 chunks with metadata
    print("\n── Sample chunks ──")
    for i in range(min(5, len(chunks))):
        print(f"\n[{i+1}] p.{metadata[i]['page']} | {metadata[i]['section_title'][:60]}")
        print(f"     {chunks[i][:200]}")
