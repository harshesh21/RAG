from langchain_text_splitters import CharacterTextSplitter
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Sample document — a paragraph about RAG
doc = """
Retrieval Augmented Generation (RAG) is a technique that enhances large language 
models by retrieving relevant information from external knowledge bases before 
generating responses. RAG systems work by first converting documents into vector 
embeddings and storing them in a vector database. When a user submits a query, 
the system retrieves the most relevant document chunks using similarity search. 
These retrieved chunks are then passed to the language model as additional context, 
allowing it to generate more accurate and grounded responses. RAG is particularly 
useful in enterprise settings where models need access to proprietary or frequently 
updated information that was not included in the original training data. The quality 
of a RAG system depends heavily on the chunking strategy used to split documents, 
the quality of embeddings, and the retrieval mechanism employed.
"""

# Fixed size splitter
fixed_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator=""
)

fixed_chunks = fixed_splitter.split_text(doc)

print(f"Fixed chunking — chunk size 200 chars\n")
print(f"Total chunks: {len(fixed_chunks)}\n")
for i, chunk in enumerate(fixed_chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars):")
    print(f"  {chunk[:100]}...")
    print()


# Recursive splitter — tries to split on paragraphs, then sentences, then words
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " ", ""]
)

recursive_chunks = recursive_splitter.split_text(doc)

print(f"Recursive chunking — chunk size 200 chars\n")
print(f"Total chunks: {len(recursive_chunks)}\n")
for i, chunk in enumerate(recursive_chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars):")
    print(f"  {chunk[:]}...")
    print()

# Token based — more accurate for LLM context windows
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=50,
    chunk_overlap=10,
    encoding_name="cl100k_base"
)

token_chunks = token_splitter.split_text(doc)

print(f"Token chunking — chunk size 50 tokens\n")
print(f"Total chunks: {len(token_chunks)}\n")
for i, chunk in enumerate(token_chunks):
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(chunk))
    print(f"Chunk {i+1} ({token_count} tokens):")
    print(f"  {chunk[:100]}...")
    print()



from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query, chunks, top_k=2):
    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    results = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]

query = "How does chunking affect RAG quality?"

print("=" * 60)
print(f"Query: '{query}'\n")

print("Fixed chunks — top 2 results:")
for chunk, score in retrieve(query, fixed_chunks):
    print(f"  {score:.3f}  {chunk[:80]}...")

print("\nRecursive chunks — top 2 results:")
for chunk, score in retrieve(query, recursive_chunks):
    print(f"  {score:.3f}  {chunk[:80]}...")

print("\nToken chunks — top 2 results:")
for chunk, score in retrieve(query, token_chunks):
    print(f"  {score:.3f}  {chunk[:80]}...")