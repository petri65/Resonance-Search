
import numpy as np
from typing import Tuple

def normalize(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        n = np.linalg.norm(X) or 1.0
        return X / n
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n==0] = 1.0
    return X / n

def expected_cosine_batched(X: np.ndarray, seed_vec: np.ndarray, sigma: float = 1.3, samples: int = 4096, seed: int = 7, batch: int = 50000) -> np.ndarray:
    """Compute E[cos(z, x)] for all x in X, z ~ N(seed_vec, sigma^2 I) in batches."""
    rng = np.random.default_rng(seed)
    P = rng.normal(loc=seed_vec, scale=sigma, size=(samples, X.shape[1]))
    Pn = normalize(P)
    Xn = normalize(X)
    out = np.empty(X.shape[0], dtype=float)
    for start in range(0, X.shape[0], batch):
        end = min(start + batch, X.shape[0])
        out[start:end] = (Xn[start:end] @ Pn.T).mean(axis=1)
    return out

def compute_scores_batched(X: np.ndarray, seed_vec: np.ndarray, gain: np.ndarray, sigma: float = 1.3, samples: int = 4096, batch: int = 50000):
    strict = (normalize(X) @ normalize(seed_vec)).ravel()
    fuzzy  = expected_cosine_batched(X, seed_vec, sigma=sigma, samples=samples, batch=batch)
    resonance = fuzzy * gain
    reveal = np.clip(resonance - strict, 0.0, None)
    return strict, fuzzy, resonance, reveal
