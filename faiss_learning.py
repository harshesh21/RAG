import faiss
import numpy as np
import time

# Simulate 10,000 documents with 384-dim embeddings (same as all-MiniLM-L6-v2)
np.random.seed(42)
d = 384
n = 10000
vectors = np.random.random((n, d)).astype('float32')
query = np.random.random((1, d)).astype('float32')

# --- Index 1: Flat (brute force) ---
index_flat = faiss.IndexFlatL2(d)
index_flat.add(vectors)
start = time.time()
D, I = index_flat.search(query, k=5)
flat_time = time.time() - start
print(f"Flat     → {flat_time*1000:.2f}ms  | top result index: {I[0][0]}")

# --- Index 2: IVFFlat (cluster-based) ---
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(vectors)
index_ivf.add(vectors)
index_ivf.nprobe = 10
start = time.time()
D, I = index_ivf.search(query, k=5)
ivf_time = time.time() - start
print(f"IVFFlat  → {ivf_time*1000:.2f}ms  | top result index: {I[0][0]}")

# --- Index 3: HNSW ---
index_hnsw = faiss.IndexHNSWFlat(d, 32)
index_hnsw.add(vectors)
start = time.time()
D, I = index_hnsw.search(query, k=5)
hnsw_time = time.time() - start
print(f"HNSW     → {hnsw_time*1000:.2f}ms  | top result index: {I[0][0]}")

# Summary
print(f"\nHNSW is {flat_time/hnsw_time:.1f}x faster than Flat")