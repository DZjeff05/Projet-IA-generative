# CareerLens AI

CareerLens AI est une application web d'analyse de profil professionnel basée sur l'intelligence artificielle. Elle permet à un utilisateur de décrire ses compétences en texte libre et d'obtenir en retour une analyse sémantique détaillée : scores par blocs de compétences, recommandations de métiers compatibles, plan de progression personnalisé et bio professionnelle générée automatiquement.

Le système repose sur trois briques principales : un moteur d'embedding SBERT pour la comparaison sémantique, un index vectoriel FAISS pour le stockage des représentations, et un pipeline RAG connecté à l'API Gemini de Google pour la génération de contenu.

---

## Prérequis

- Python 3.10 ou supérieur
- pip
- Git
- Un compte Google pour obtenir une clé API Gemini (gratuit)

---

## Obtenir une clé API Gemini

1. Rendez-vous sur [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Connectez-vous avec votre compte Google
3. Cliquez sur **Create API key**
4. Copiez la clé générée — vous en aurez besoin à l'étape suivante

Le modèle utilisé est `gemini-2.5-flash`, disponible gratuitement dans les limites du quota standard.

---

## Installation

**1. Cloner le projet**

```bash
git clone https://github.com/votre-utilisateur/Projet_IA_Generative.git
cd Projet_IA_Generative
```

**2. Créer et activer un environnement virtuel**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Installer les dépendances**

```bash
pip install -r requirements.txt
```

Le téléchargement du modèle SBERT (`all-mpnet-base-v2`) se fait automatiquement au premier lancement. Il pèse environ 420 Mo.

---

## Configuration de la clé API

Créez un fichier `.env` à la racine du projet (au même niveau que `requirements.txt`) :

```
GEMINI_API_KEY=votre_clé_ici
```

Remplacez `votre_clé_ici` par la clé copiée depuis Google AI Studio. Le fichier doit ressembler à ceci :

```
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Ce fichier est listé dans `.gitignore` et ne sera jamais versionné. Ne le partagez pas.

---

## Lancer l'application

```bash
streamlit run ui/app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

Au premier lancement, le modèle SBERT est téléchargé depuis Hugging Face. Les lancements suivants sont instantanés grâce au cache local.

---

## Structure du projet

```
Projet_IA_Generative/
├── configs/
│   ├── api_keys.py          # Lecture de la clé depuis .env
│   └── settings.py          # Chemins et constantes globales
├── data/
│   └── referentiel_competences.json   # Référentiel métier (blocs + compétences + métiers)
├── embedding_engine/
│   ├── embedder.py          # API publique d'embedding
│   └── sbert_model.py       # Singleton SBERT
├── llm_engine/
│   └── generator.py         # Appels Gemini + cache MD5
├── rag_engine/
│   ├── context_builder.py   # Construction du contexte RAG
│   └── scorer.py            # Scoring SBERT + job matching pondéré
├── vector_store/
│   └── faiss_index.py       # VectorStore FAISS (768 dimensions)
├── ui/
│   └── app.py               # Interface Streamlit
├── cache/
│   └── llm_cache.json       # Cache des réponses Gemini (généré automatiquement)
├── .env                     # Votre clé API (à créer, non versionné)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Remarques

- Le cache LLM (`cache/llm_cache.json`) évite de rappeler l'API Gemini pour des profils déjà analysés. Si vous souhaitez forcer une nouvelle génération, supprimez ce fichier ou videz son contenu.
- Le modèle SBERT est chargé une seule fois en mémoire par session. Ne relancez pas l'application entre deux analyses, utilisez simplement le bouton d'analyse autant de fois que nécessaire.
- Si vous rencontrez une erreur liée aux imports (`ModuleNotFoundError`), assurez-vous que l'environnement virtuel est bien activé avant de lancer Streamlit.
