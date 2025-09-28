
#!/usr/bin/env python3
import os, glob, numpy as np, pandas as pd
from tqdm import tqdm
from wiki_resonance.embed import BatchEmbedder

def main(data_dir="wiki_data", out_dir="wiki_embeddings", model="sentence-transformers/all-MiniLM-L6-v2", batch_size=256):
    os.makedirs(out_dir, exist_ok=True)
    be = BatchEmbedder(model)
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "wiki_*.parquet")))
    for sp in shard_paths:
        df = pd.read_parquet(sp)
        texts = df["text"].astype(str).tolist()
        X = be.encode_texts(texts, batch_size=batch_size, normalize=True)
        np.save(os.path.join(out_dir, os.path.basename(sp).replace(".parquet", ".npy")), X)
        df[["id","title","url"]].to_parquet(os.path.join(out_dir, os.path.basename(sp).replace(".parquet", ".meta.parquet")))
        print("Embedded:", sp, "→", X.shape)
    print("Done. Embeddings in:", out_dir)

if __name__ == "__main__":
    main()
