
#!/usr/bin/env python3
import os, glob, numpy as np, pandas as pd
from tqdm import tqdm
from wiki_resonance.embed import BatchEmbedder
from wiki_resonance.resonance import compute_scores_batched
from wiki_resonance.pagerank import pagerank_gain
import scipy.sparse as sp

def main(seed="Elephant", emb_dir="wiki_embeddings", index_dir="wiki_index", sigma=1.3, samples=4096, topk=50):
    print("[1/4] Loading embeddings shards (meta only)")
    meta_files = sorted(glob.glob(os.path.join(emb_dir, "wiki_*.meta.parquet")))
    emb_files  = [f.replace(".meta.parquet", ".npy") for f in meta_files]
    metas = [pd.read_parquet(f)[["id","title","url"]] for f in meta_files]
    meta = pd.concat(metas, ignore_index=True)

    print("[2/4] Loading kNN graph")
    An = sp.load_npz(os.path.join(index_dir, "wiki_knn_rownorm.npz"))
    indptr, indices, data = An.indptr, An.indices, An.data
    print("  Nodes:", An.shape[0])

    print("[3/4] Computing spectral gain via PageRank (low-frequency emphasis)")
    gain = pagerank_gain(indptr, indices, data, alpha=0.15, iters=60)

    print("[4/4] Embedding seed and computing resonance in batches")
    be = BatchEmbedder()
    seed_vec = be.encode_seed(seed)
    # stream through shards to compute scores and write partial results
    rows = []
    offset = 0
    for ef, mf in tqdm(zip(emb_files, meta_files), total=len(emb_files), desc="Scoring"):
        X = np.load(ef)
        strict, fuzzy, resonance, reveal = compute_scores_batched(X, seed_vec, gain[offset:offset+len(X)], sigma=sigma, samples=samples, batch=50000)
        m = pd.read_parquet(mf)
        part = pd.DataFrame({
            "global_id": np.arange(offset, offset+len(X)),
            "title": m["title"].values,
            "url": m["url"].values,
            "strict": strict,
            "fuzzy": fuzzy,
            "gain": gain[offset:offset+len(X)],
            "resonance": resonance,
            "reveal": reveal
        })
        rows.append(part)
        offset += len(X)
    df = pd.concat(rows, ignore_index=True).sort_values("reveal", ascending=False)
    out_csv = os.path.join(index_dir, f"resonance_{seed.replace(' ','_')}.csv")
    df.head(topk).to_csv(out_csv, index=False)
    print("Top results saved to:", out_csv)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=str, default="Elephant")
    ap.add_argument("--emb_dir", type=str, default="wiki_embeddings")
    ap.add_argument("--index_dir", type=str, default="wiki_index")
    ap.add_argument("--sigma", type=float, default=1.3)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()
    main(seed=args.seed, emb_dir=args.emb_dir, index_dir=args.index_dir, sigma=args.sigma, samples=args.samples, topk=args.topk)
