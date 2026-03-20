# rag_engine/context_builder.py

def extract_skill_gaps(block_scores, threshold=0.55):
    """
    Identifie les blocs et compétences faibles
    """

    weak_blocks = []
    weak_skills = []

    for block in block_scores:

        if block["average_score"] < threshold:
            weak_blocks.append(block["block_name"])

            for comp in block["matched_competencies"]:
                if comp["score"] < threshold:
                    weak_skills.append(comp["competency"])

    return {
        "weak_blocks": weak_blocks,
        "weak_skills": list(set(weak_skills))
    }


def build_rag_context(user_text, block_scores):
    """
    Construit le contexte envoyé au LLM
    """

    gaps = extract_skill_gaps(block_scores)

    context = f"""
USER PROFILE:
{user_text}

WEAK COMPETENCY BLOCKS:
{', '.join(gaps['weak_blocks'])}

SKILLS TO IMPROVE:
{', '.join(gaps['weak_skills'])}
"""

    return context, gaps