from typing import List
from embedding_engine.sbert_model import get_sbert_model

def embed_texts(texts: List[str]):
    model = get_sbert_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def embed_single_text(text: str):
    model = get_sbert_model()
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]