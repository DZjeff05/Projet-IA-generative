import json
import hashlib
from pathlib import Path
import google.generativeai as genai

from configs.api_keys import GEMINI_API_KEY


# =============================
# GEMINI CONFIG
# =============================

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


# =============================
# CACHE CONFIG
# =============================

CACHE_FILE = Path("cache/llm_cache.json")
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


# =============================
# CACHE UTILITIES
# =============================

def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def hash_context(context: str):
    return hashlib.md5(context.encode("utf-8")).hexdigest()


# =============================
# CORE GENERATION WITH CACHE
# =============================

def generate_with_cache(prompt: str):

    cache = load_cache()
    key = hash_context(prompt)

    if key in cache:
        return cache[key]

    response = model.generate_content(prompt)
    text = response.text

    cache[key] = text
    save_cache(cache)

    return text


# =============================
# RAG — PROGRESSION PLAN
# =============================

def generate_progression_plan(context: str):

    prompt = f"""
You are a career mentor specialized in data science and AI roles.
Use ONLY the context below. Do not invent skills or experience.
Always respond in French.

CONTEXT: {context}

Write a structured progression plan with the following sections:

**Points forts actuels**
2-3 phrases identifiant les compétences bien couvertes et ce qu'elles permettent concrètement.

**Lacunes identifiées**
2-3 phrases décrivant les blocs faibles et pourquoi ils freinent l'évolution vers les métiers cibles.

**Axes d'apprentissage prioritaires**
Liste de 3 à 4 sujets à travailler, avec pour chacun une ressource ou approche concrète (cours, projet, outil).

**Certifications recommandées**
Propose 2 à 3 certifications pertinentes et accessibles en lien avec le profil. Précise l'organisme et le niveau.

**Plan d'action 30 / 60 / 90 jours**
3 étapes courtes et actionnables, une par palier temporel.

Chaque section doit être concise mais utile. Pas d'introduction, pas de conclusion.
"""

    return generate_with_cache(prompt)


# =============================
# RAG — PROFESSIONAL BIO
# =============================

def generate_professional_bio(context: str):

    prompt = f"""
You are a profile writer. Use ONLY the context below. Do not invent skills.
Always respond in French, regardless of the input language.

CONTEXT: {context}

Write a 3-sentence professional bio:
1. Current expertise and technical stack
2. Key strength or differentiator
3. Career positioning

Tone: confident, concise. No filler phrases.
"""

    return generate_with_cache(prompt)