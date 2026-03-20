from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "outputs"

REFERENCE_FILE = DATA_DIR / "referentiel_competences.json"

SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

TOP_K_COMPETENCIES = 5
TOP_K_JOBS = 3

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)