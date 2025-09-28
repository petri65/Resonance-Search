#!/usr/bin/env python3
"""
Resonance Finder v1
- Interactive script that crawls from seed URLs (no search engine)
- Extracts text, embeds documents + seed text
- Computes resonance (expected cosine × spectral gain)
- Ranks URLs by REVEAL score
"""

import sys

# --- Dependency check wrapper ---
REQUIRED = [
    "trafilatura",
    "requests",
    "bs4",
    "numpy",
    "scipy",
    "pandas",
    "tqdm",
    "sentence_transformers",
    "sklearn"
]

missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg.split(".")[0])  # import by top-level
    except ImportError:
        missing.append(pkg)

if missing:
    print("\n[ERROR] Missing required packages:")
    for m in missing:
        print(f"  - {m}")
    print("\nPlease install them with:")
    print("  python3 -m pip install -r requirements.txt")
    sys.exit(1)

# --- Normal imports after check ---
import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import trafilatura
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


# === Helper functions (same as before) ===

def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        n = np.linalg.norm(v) or 1.0
        return v / n
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return v / n


def polite_get(url: str, timeout: int = 12) -> str:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "resonance-client/0.1 (+research)"}
        )
        if resp.status_code != 200:
            return ""
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return ""
        return resp.text
    except Exception:
        return ""


def extract_text(html: str, max_chars: int = 200000) -> str:
    if not html:
        return ""
    txt = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    return txt[:max_chars].strip()


def crawl_from_seeds(seeds, max_pages=40, same_domain_only=True, sleep=0.5, min_len=200):
    seen = set()
    out_urls, out_texts = [], []
    allowed = {urlparse(s).netloc for s in seeds} if same_domain_only else None
    q = deque(seeds)
    pbar = tqdm(total=max_pages, desc="Crawling", ncols=90)

    while q and len(out_urls) < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        html = polite_get(url)
        if not html:
            continue

        text = extract_text(html)
        if text and len(text) >= min_len:
            out_urls.append(url)
            out_texts.append(text)
            pbar.update(1)

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a")[:25]:
            href = a.get("href")
            if not href:
                continue
            nxt = urljoin(url, href)
            if not nxt.startswith("http"):
                continue
            host = urlparse(nxt).netloc
            if same_domain_only and host not in allowed:
                continue
            if nxt not in seen:
                q.append(nxt)

        time.sleep(sleep)

    pbar.close()
    return list(zip(out_urls, out_texts))


def build_knn_graph(embeddings: np.ndarray, k=15) -> csr_matrix:
    X = normalize(embeddings)
    if len(X) == 0:
        return csr_matrix((0, 0))
    k = max(1, min(k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)
    rows, cols, vals = [], [], []
    for i in range(len(X)):
        for j, d in zip(idx[i][1:], dist[i][1:]):
            sim = 1.0 - float(d)
            if sim < 0:
                sim = 0.0
            rows.append(i); cols.append(j); vals.append(sim)
    A = csr_matrix((vals, (rows, cols)), shape=(len(X), len(X)))
    return A.maximum(A.T)


def spectral_gain(A: csr_matrix, k=48) -> np.ndarray:
    if A.shape[0] == 0:
        return np.zeros((0,))
    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    Dm12 = diags(1.0 / np.sqrt(deg))
    L = diags(np.ones(A.shape[0])) - (Dm12 @ A @ Dm12)
    k = min(k, max(1, A.shape[0] - 2))
    vals, vecs = eigsh(L, k=k, which="SM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    denom = 1.0 + vals
    weights = (vecs ** 2) / denom[None, :]
    g = weights.sum(axis=1)
    return g / (g.mean() + 1e-12)


def expected_cosine(embeddings, mu, sigma=1.3, samples=2048, seed=7):
    rng = np.random.default_rng(seed)
    Xn = normalize(embeddings)
    mu = np.asarray(mu, dtype=float)
    P = rng.normal(loc=mu, scale=sigma, size=(samples, mu.shape[0]))
    Pn = normalize(P)
    sims = Xn @ Pn.T
    return sims.mean(axis=1)


def resonance_scores(embeddings, seed_vec, A, sigma=1.3, samples=2048):
    strict = (normalize(embeddings) @ normalize(seed_vec)).ravel()
    fuzzy = expected_cosine(embeddings, seed_vec, sigma=sigma, samples=samples)
    gain = spectral_gain(A, k=48)
    resonance = fuzzy * gain
    reveal = np.clip(resonance - strict, 0.0, None)
    return strict, resonance, reveal, gain


# === Main ===
def main():
    print("\n=== Resonance Finder v1 ===")
    seed_text = input("Enter a seed word/phrase: ").strip()
    if not seed_text:
        print("No seed provided. Exiting.")
        return

    seeds_raw = input("Enter seed URLs (comma-separated) [default: Wikipedia main page]: ").strip()
    seed_urls = [s.strip() for s in seeds_raw.split(",") if s.strip().startswith("http")] or [
        "https://en.wikipedia.org/wiki/Main_Page"
    ]

    items = crawl_from_seeds(seed_urls, max_pages=30)
    if not items:
        print("No pages fetched.")
        return

    urls = [u for (u, _t) in items]
    texts = [t for (_u, t) in items]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    seed_vec = model.encode([seed_text], normalize_embeddings=True, convert_to_numpy=True)[0]

    A = build_knn_graph(X, k=15)
    strict, resonance, reveal, gain = resonance_scores(X, seed_vec, A)

    df = pd.DataFrame({"url": urls, "strict": strict, "resonance": resonance, "reveal": reveal, "gain": gain})
    df = df.sort_values("reveal", ascending=False)

    print("\n=== Top resonating URLs ===")
    for _, row in df.head(10).iterrows():
        print(f"{row['reveal']:.3f}  {row['url']}")

    df.to_csv("resonance_results.csv", index=False)
    print("\nSaved all results to resonance_results.csv")


if __name__ == "__main__":
    main()
