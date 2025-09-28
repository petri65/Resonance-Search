
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Iterable, List

class BatchEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)
    def encode_texts(self, texts: List[str], batch_size: int = 256, normalize: bool = True) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize, convert_to_numpy=True)
    def encode_seed(self, seed: str, normalize: bool = True) -> np.ndarray:
        return self.model.encode([seed], normalize_embeddings=normalize, convert_to_numpy=True)[0]
