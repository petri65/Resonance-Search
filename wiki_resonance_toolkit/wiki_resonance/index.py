
import hnswlib, numpy as np, os, json

class HNSWIndex:
    def __init__(self, dim: int, space: str = "cosine"):
        self.dim = dim
        self.index = hnswlib.Index(space=space, dim=dim)
        self._init = False
    def build(self, X: np.ndarray, ef_construction: int = 200, M: int = 48, ids: np.ndarray | None = None):
        num = X.shape[0]
        self.index.init_index(max_elements=num, ef_construction=ef_construction, M=M)
        if ids is None:
            ids = np.arange(num, dtype=np.int64)
        self.index.add_items(X, ids)
        self.index.set_ef(200)
        self._init = True
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.index.save_index(path)
        with open(path + ".meta.json", "w") as f:
            json.dump({"dim": self.dim}, f)
    @staticmethod
    def load(path: str):
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        idx = HNSWIndex(dim=meta["dim"])
        idx.index.load_index(path)
        idx._init = True
        return idx
    def knn(self, X: np.ndarray, k: int = 30):
        return self.index.knn_query(X, k=k)
