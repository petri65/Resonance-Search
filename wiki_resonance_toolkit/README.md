
# Wikipedia Resonance Toolkit (Entropy Dissipation on the whole Wikipedia)

This toolkit computes **resonance** over (the whole) Wikipedia without any search engine:
- Loads a Wikipedia dump via the Hugging Face `wikimedia/wikipedia` dataset
- Embeds all pages (batched, normalized)
- Builds a scalable **HNSW** kNN graph
- Approximates **spectral gain** with **PageRank** on the kNN graph (low-frequency/global modes)
- Answers queries using **expected cosine** under Gaussian jitter (fuzziness) and outputs top resonators by **REVEAL**

## Install
```bash
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

1) **Download/prepare Wikipedia**
```bash
python prepare_wiki.py --out_dir wiki_data --shard_size 50000
# uses lang_config=20231101.en by default; adjust if needed
```

2) **Embed shards**
```bash
python embed_wiki.py --data_dir wiki_data --out_dir wiki_embeddings --model sentence-transformers/all-MiniLM-L6-v2
```

3) **Build index + kNN graph**
```bash
python build_graph.py --emb_dir wiki_embeddings --out_dir wiki_index --k 30
```

4) **Query resonance**
```bash
python query_resonance.py --seed "Elephant" --emb_dir wiki_embeddings --index_dir wiki_index --sigma 1.3 --samples 4096 --topk 50
# results saved to wiki_index/resonance_Elephant.csv
```

## Notes
- **Scale**: English Wikipedia is big; run this on a machine with ample disk/RAM (you can start with a subset by using fewer shards).
- **Spectral gain**: we use PageRank as a robust, scalable proxy for low-frequency spectral emphasis. You can swap with a heat-kernel or Lanczos eigensolver if your resources allow.
- **Fuzziness (σ)**: increase to pull in semi-identical concepts; too large will add noise.
- **Safety**: This pipeline is read-only over Wikipedia content; no external crawling/ranking.

## Outputs
CSV with columns:
- `title`, `url`
- `strict` (cosine to exact seed), `fuzzy` (expected cosine), `gain` (PageRank), `resonance` = `fuzzy×gain`
- `reveal` = `max(resonance − strict, 0)` — what becomes salient only under fuzziness + global modes.
