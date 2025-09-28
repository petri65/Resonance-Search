
#!/usr/bin/env python3
import os, pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def main(lang_config="20231101.en", out_dir="wiki_data", shard_size=50000):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", lang_config, split="train")
    total = len(ds)
    shard = []
    shard_id = 0
    for i, row in tqdm(enumerate(ds), total=total, desc="Downloading Wikipedia"):
        title = row.get("title") or ""
        url = row.get("url") or ""
        text = row.get("text") or ""
        shard.append({"id": i, "title": title, "url": url, "text": text})
        if len(shard) >= shard_size:
            pd.DataFrame(shard).to_parquet(os.path.join(out_dir, f"wiki_{shard_id:05d}.parquet"))
            shard = []; shard_id += 1
    if shard:
        pd.DataFrame(shard).to_parquet(os.path.join(out_dir, f"wiki_{shard_id:05d}.parquet"))
    print("Done. Shards in:", out_dir)

if __name__ == "__main__":
    main()
