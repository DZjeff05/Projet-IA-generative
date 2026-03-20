import json
from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity

from configs.settings import REFERENCE_FILE
from embedding_engine.embedder import embed_single_text, embed_texts
from vector_store.faiss_index import VectorStore


# =====================================================
# LOAD REFERENCE DATA
# =====================================================

def load_reference() -> Dict:
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# VECTOR STORE INITIALIZATION
# =====================================================

vector_store = VectorStore(768)


# =====================================================
# BLOCK WEIGHTS
# =====================================================

# Blocs techniques coeur = poids élevé
# Blocs support/transversal = poids plus faible
BLOCK_WEIGHTS = {
    "Machine Learning":              1.5,
    "Natural Language Processing":   1.4,
    "Deep Learning":                 1.4,
    "Data Analysis":                 1.3,
    "Data Engineering":              1.3,
    "Software Engineering":          1.2,
    "Computer Vision":               1.2,
    "Cloud & DevOps":                1.1,
    "Business Intelligence":         1.0,
    "Project Management":            0.8,  # compétence transversale, moins discriminante
}

DEFAULT_WEIGHT = 1.0


# =====================================================
# COMPUTE COMPETENCY BLOCK SCORES
# =====================================================

def compute_block_scores(user_text: str) -> List[Dict]:
    reference = load_reference()
    user_embedding = embed_single_text(user_text)
    results = []
    raw_combined_scores = []

    for block in reference["competency_blocks"]:
        competencies = block["competencies"]
        comp_embeddings = embed_texts(competencies)
        sims = cosine_similarity([user_embedding], comp_embeddings)[0]

        avg_score = float(sims.mean())
        max_score = float(sims.max())

        # Formule combinée : moyenne + max pour valoriser les bons matches
        combined_score = 0.5 * avg_score + 0.5 * max_score
        raw_combined_scores.append(combined_score)

        metadata_list = [
            {
                "block_id": block["block_id"],
                "block_name": block["block_name"]
            }
            for _ in competencies
        ]

        vector_store.add_embeddings(
            embeddings=comp_embeddings,
            texts=competencies,
            metadata=metadata_list
        )

        results.append({
            "block_id": block["block_id"],
            "block_name": block["block_name"],
            "average_score": round(combined_score, 4),
            "max_score": round(max_score, 4),
            "raw_avg": round(avg_score, 4),
            "weight": BLOCK_WEIGHTS.get(block["block_name"], DEFAULT_WEIGHT),
            "matched_competencies": sorted(
                [
                    {"competency": comp, "score": round(float(score), 4)}
                    for comp, score in zip(competencies, sims)
                ],
                key=lambda x: x["score"],
                reverse=True
            )[:5]
        })

    # Normalisation min-max pour étaler les scores sur [0, 1]
    min_s = min(raw_combined_scores)
    max_s = max(raw_combined_scores)
    score_range = max_s - min_s if max_s != min_s else 1.0

    for r in results:
        normalized = (r["average_score"] - min_s) / score_range
        r["average_score"] = round(normalized, 4)

    results.sort(key=lambda x: x["average_score"], reverse=True)
    return results


# =====================================================
# JOB MATCHING SCORES — AVEC PONDÉRATION
# =====================================================

def compute_job_scores(block_scores: List[Dict]) -> List[Dict]:
    reference = load_reference()

    block_score_map = {
        block["block_name"]: {
            "score": block["average_score"],
            "weight": block.get("weight", DEFAULT_WEIGHT)
        }
        for block in block_scores
    }

    job_results = []

    for job in reference["job_profiles"]:
        required_blocks = job["required_blocks"]

        weighted_sum = 0.0
        total_weight = 0.0

        for block_name in required_blocks:
            block_data = block_score_map.get(
                block_name,
                {"score": 0.0, "weight": DEFAULT_WEIGHT}
            )
            w = block_data["weight"]
            s = block_data["score"]
            weighted_sum += w * s
            total_weight += w

        coverage = weighted_sum / total_weight if total_weight > 0 else 0.0

        job_results.append({
            "job_id": job["job_id"],
            "job_title": job["job_title"],
            "coverage_score": round(coverage, 4),
            "required_blocks": required_blocks
        })

    job_results.sort(key=lambda x: x["coverage_score"], reverse=True)
    return job_results


# =====================================================
# HEATMAP DATA GENERATION
# =====================================================

def generate_skill_heatmap(block_scores: List[Dict]) -> Dict:
    heatmap = {"labels": [], "values": []}
    for block in block_scores:
        heatmap["labels"].append(block["block_name"])
        heatmap["values"].append(block["average_score"])
    return heatmap


# =====================================================
# FULL SCORING PIPELINE
# =====================================================

def score_profile(user_text: str) -> Dict:
    block_scores = compute_block_scores(user_text)
    job_scores = compute_job_scores(block_scores)
    heatmap = generate_skill_heatmap(block_scores)

    return {
        "block_scores": block_scores,
        "job_matches": job_scores,
        "heatmap": heatmap
    }