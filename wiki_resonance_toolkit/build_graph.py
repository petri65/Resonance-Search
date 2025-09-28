
#!/usr/bin/env python3
import os, glob, numpy as np, pandas as pd
from tqdm import tqdm
from wiki_resonance.index import HNSWIndex

def main(emb_dir="wiki_embeddings", out_dir="wiki_index", M=48, efc=200, efq=200, k=30):
    os.makedirs(out_dir, exist_ok=True)
    # Build index shard-by-shard
    first = sorted(glob.glob(os.path.join(emb_dir, "wiki_*.npy")))[0]
    X0 = np.load(first)
    dim = X0.shape[1]
    idx = HNSWIndex(dim=dim)
    # Count total vectors
    files = sorted(glob.glob(os.path.join(emb_dir, "wiki_*.npy")))
    total = sum(np.load(f).shape[0] for f in files)
    idx.index.init_index(max_elements=total, ef_construction=efc, M=M)
    cur = 0
    for f in tqdm(files, desc="Adding to HNSW"):
        X = np.load(f)
        ids = np.arange(cur, cur + len(X), dtype=np.int64)
        idx.index.add_items(X, ids)
        cur += len(X)
    idx.index.set_ef(efq)
    idx.save(os.path.join(out_dir, "wiki_hnsw.bin"))
    print("Saved HNSW to", out_dir)

    # Build kNN edges in batches (to save memory)
    K = k
    rows, cols = [], []
    for f in tqdm(files, desc="Query kNN"):
        X = np.load(f)
        I, D = idx.knn(X, k=K+1)  # include self
        for row_base, neigh in enumerate(I):
            gid = len(rows)  # temporary; will fix with offset below
        # We'll re-query with offsets to avoid confusion
    # Re-do with proper offsets
    rows, cols = [], []
    start = 0
    for f in tqdm(files, desc="kNN with offsets"):
        X = np.load(f)
        I, D = idx.knn(X, k=K+1)
        for i, neigh in enumerate(I):
            gi = start + i
            for j in neigh[1:]:  # skip self
                rows.append(gi); cols.append(int(j))
        start += len(X)

    # Save CSR as npz (row-normalized weights = 1/K)
    import scipy.sparse as sp
    data = np.ones(len(rows), dtype=float) / K
    n = total
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n)).maximum(sp.csr_matrix((data, (cols, rows)), shape=(n, n)))
    # Row-normalize
    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg==0.0] = 1.0
    Dinv = sp.diags(1.0/deg)
    An = Dinv @ A
    sp.save_npz(os.path.join(out_dir, "wiki_knn_rownorm.npz"), An)
    print("Saved kNN graph:", os.path.join(out_dir, "wiki_knn_rownorm.npz"))

if __name__ == "__main__":
    main()
