import faiss
import numpy as np
import pickle
from pathlib import Path


# =============================
# PATH CONFIG
# =============================

INDEX_PATH = Path("vector_store/faiss.index")
META_PATH = Path("vector_store/faiss_meta.pkl")


# =============================
# VECTOR STORE CLASS
# =============================

class VectorStore:
    """
    FAISS Vector Database for AISCA RAG system.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.metadata = []

    def add(self, embeddings, texts, metadata=None):
        embeddings = np.array(embeddings).astype("float32")
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings)
        self.texts.extend(texts)
        if metadata is None:
            metadata = [{} for _ in texts]
        self.metadata.extend(metadata)

    def add_embeddings(self, embeddings, texts, metadata=None):
        self.add(embeddings, texts, metadata)

    def search(self, query_embedding, k=5):
        if self.index.ntotal == 0:
            return []
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx]
                })
        return results

    def save(self):
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)
        print("FAISS index saved")

    def load(self):
        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                self.texts = data
                self.metadata = [{} for _ in data]
            else:
                self.texts = data["texts"]
                self.metadata = data["metadata"]
            print("FAISS index loaded")
        else:
            print("No FAISS index found — starting fresh")

    def size(self):
        return self.index.ntotal