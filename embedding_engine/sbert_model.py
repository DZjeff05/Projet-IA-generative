from sentence_transformers import SentenceTransformer
from configs.settings import SBERT_MODEL_NAME

_model = None

def get_sbert_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(SBERT_MODEL_NAME)
    return _model