
import numpy as np

def pagerank_gain(indptr, indices, data, alpha: float = 0.15, iters: int = 50):
    """Approximate low-frequency spectral gain via PageRank centrality on kNN graph.
    Inputs are CSR components (indptr, indices, data) of row-normalized adjacency.
    Returns a per-node gain vector normalized to mean=1.
    """
    n = len(indptr) - 1
    pr = np.ones(n, dtype=float) / n
    teleport = (1.0 - alpha) / n
    for _ in range(iters):
        new = np.zeros_like(pr)
        for i in range(n):
            row_start, row_end = indptr[i], indptr[i+1]
            js = indices[row_start:row_end]
            ws = data[row_start:row_end]
            if ws.size == 0:
                continue
            # distribute pr[i] to neighbors
            new[js] += alpha * pr[i] * ws
        pr = new + teleport
        pr /= pr.sum()
    return pr / (pr.mean() + 1e-12)
