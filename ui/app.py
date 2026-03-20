import sys
from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =============================
# FIX IMPORT PATH
# =============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rag_engine.scorer import compute_block_scores, compute_job_scores
from rag_engine.context_builder import build_rag_context
from llm_engine.generator import generate_progression_plan, generate_professional_bio

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="CareerLens AI", page_icon="◈", layout="wide")

# =============================
# CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #09090f;
    --surface:   #111118;
    --border:    rgba(255,255,255,0.06);
    --gold:      #c9a84c;
    --gold-dim:  rgba(201,168,76,0.15);
    --gold-glow: rgba(201,168,76,0.08);
    --text:      #e8e4dc;
    --muted:     rgba(232,228,220,0.4);
    --radius:    14px;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 2px; }

/* ── hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem !important; max-width: 1100px; margin: 0 auto; }

/* ── hero ── */
.hero {
    position: relative;
    padding: 56px 52px 48px;
    margin-bottom: 48px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 90% 10%, rgba(201,168,76,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 10% 90%, rgba(201,168,76,0.04) 0%, transparent 50%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    font-weight: 400;
    color: var(--text);
    line-height: 1.1;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.hero-title em {
    font-style: italic;
    color: var(--gold);
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    font-weight: 300;
    max-width: 480px;
    line-height: 1.7;
    margin: 14px 0 0;
}
.hero-deco {
    position: absolute;
    right: 52px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.04;
    font-family: 'DM Serif Display', serif;
    color: var(--gold);
    pointer-events: none;
    user-select: none;
}

/* ── section label ── */
.label {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
}
.card-score {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 400;
    color: var(--gold);
    line-height: 1;
}
.card-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 8px;
}
.card-level {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--text);
    margin-top: 4px;
}

/* ── progress bars ── */
.prog-wrap { margin-bottom: 10px; }
.prog-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 5px;
}
.prog-track {
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--gold), #e8c87a);
    border-radius: 2px;
    transition: width 0.8s cubic-bezier(.16,1,.3,1);
}

/* ── competency tag ── */
.tag {
    display: inline-block;
    background: var(--gold-dim);
    border: 1px solid rgba(201,168,76,0.2);
    color: var(--gold);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin: 3px 3px 3px 0;
    font-weight: 400;
}

/* ── bio box ── */
.bio-box {
    background: var(--gold-glow);
    border-left: 2px solid var(--gold);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 20px 24px;
    font-size: 0.95rem;
    line-height: 1.8;
    color: var(--text);
    font-weight: 300;
}

/* ── plan box ── */
.plan-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    font-size: 0.9rem;
    line-height: 1.9;
    color: var(--text);
    font-weight: 300;
    white-space: pre-wrap;
}

/* ── inputs ── */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(201,168,76,0.3) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.05) !important;
}
.stSlider [data-baseweb="slider"] { margin-top: 4px; }
div[data-baseweb="slider"] div[role="slider"] {
    background: var(--gold) !important;
    border-color: var(--gold) !important;
}

/* ── button ── */
.stButton > button {
    background: var(--gold) !important;
    color: #09090f !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 13px 36px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.06em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* ── expander ── */
details {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
summary { color: var(--text) !important; font-size: 0.88rem !important; }

/* ── dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }

/* ── multiselect ── */
[data-baseweb="tag"] {
    background: var(--gold-dim) !important;
    border-color: rgba(201,168,76,0.25) !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HERO
# =============================
st.markdown("""
<div class="hero">
    <div class="hero-deco">◈</div>
    <div class="hero-eyebrow">◈ Intelligence Sémantique & Génératif</div>
    <h1 class="hero-title">Career<em>Lens</em> AI</h1>
    <p class="hero-sub">
        Positionnement métier précis basé sur l'analyse sémantique SBERT,
        enrichie par retrieval augmenté et génération Gemini.
    </p>
</div>
""", unsafe_allow_html=True)

# =============================
# INPUT
# =============================
st.markdown('<div class="label">Votre profil</div>', unsafe_allow_html=True)
user_text = st.text_area(
    label="",
    height=160,
    placeholder="Décrivez vos expériences, projets et compétences. Ex : J'ai conçu des pipelines ETL en Python, entraîné des modèles de classification et déployé des APIs REST avec FastAPI..."
)

st.markdown("<br>", unsafe_allow_html=True)

# =============================
# SELF ASSESSMENT
# =============================
st.markdown('<div class="label">Auto-évaluation</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    python_skill   = st.slider("🐍 Python / Data Analysis", 0, 5, 2)
    ml_skill       = st.slider("🤖 Machine Learning", 0, 5, 1)
    dl_skill       = st.slider("🧠 Deep Learning", 0, 5, 1)
    cv_skill       = st.slider("👁️ Computer Vision", 0, 5, 0)
with col2:
    nlp_skill      = st.slider("💬 NLP", 0, 5, 1)
    de_skill       = st.slider("⚙️ Data Engineering", 0, 5, 1)
    cloud_skill    = st.slider("☁️ Cloud & DevOps", 0, 5, 1)
with col3:
    se_skill       = st.slider("🛠️ Software Engineering", 0, 5, 2)
    bi_skill       = st.slider("📊 Business Intelligence", 0, 5, 1)
    pm_skill       = st.slider("📋 Project Management", 0, 5, 1)

st.markdown("<br>", unsafe_allow_html=True)

# =============================
# SKILLS
# =============================
st.markdown('<div class="label">Compétences maîtrisées</div>', unsafe_allow_html=True)
selected_skills = st.multiselect(
    label="",
    options=[
        # Data Analysis
        "data visualization", "statistics", "exploratory data analysis",
        "pandas data manipulation", "data cleaning",
        # Machine Learning
        "classification models", "regression models",
        "feature engineering", "model evaluation", "hyperparameter tuning",
        # Deep Learning
        "neural network training", "transfer learning",
        "convolutional neural networks", "recurrent neural networks",
        # NLP
        "tokenization", "transformers", "word embeddings",
        "text classification", "semantic analysis",
        # Data Engineering
        "ETL pipelines", "SQL querying", "Apache Spark processing",
        "workflow orchestration", "data warehousing",
        # Software Engineering
        "version control with Git", "API design",
        "object oriented programming", "unit testing", "CI/CD pipelines",
        # Computer Vision
        "object detection", "image classification",
        "openCV programming", "image segmentation",
        # Business Intelligence
        "dashboard creation", "power BI visualization",
        "KPI definition", "data storytelling",
        # Cloud & DevOps
        "docker containerization", "cloud deployment",
        "kubernetes orchestration", "infrastructure as code",
        # Project Management
        "agile methodology", "scrum workflow", "risk management"
    ]
)

st.markdown("<br><br>", unsafe_allow_html=True)

# =============================
# BUTTON
# =============================
if st.button("◈  Analyser mon profil"):

    if not user_text.strip():
        st.warning("Veuillez renseigner une description avant l'analyse.")
    else:
        with st.spinner("Analyse sémantique en cours…"):
            extra_context = " ".join(selected_skills)
            enriched_text = (
                    user_text
                    + f" Python and data analysis level {python_skill}/5"
                    + f" Machine Learning level {ml_skill}/5"
                    + f" Deep Learning level {dl_skill}/5"
                    + f" Computer Vision level {cv_skill}/5"
                    + f" NLP level {nlp_skill}/5"
                    + f" Data Engineering level {de_skill}/5"
                    + f" Cloud and DevOps level {cloud_skill}/5"
                    + f" Software Engineering level {se_skill}/5"
                    + f" Business Intelligence level {bi_skill}/5"
                    + f" Project Management level {pm_skill}/5 "
                    + extra_context
            )
            block_scores = compute_block_scores(enriched_text)
            job_scores   = compute_job_scores(block_scores)
            global_score = np.mean([b["average_score"] for b in block_scores])

            if global_score >= 0.65:
                level = "Profil Avancé"
                level_icon = "🟢"
            elif global_score >= 0.38:
                level = "Profil Intermédiaire"
                level_icon = "🟡"
            else:
                level = "Profil Débutant"
                level_icon = "🔴"

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Score global ──
        st.markdown('<div class="label">Score Global</div>', unsafe_allow_html=True)

        top_job = job_scores[0] if job_scores else None

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown(f"""
                    <div class="card">
                        <div class="card-score">{global_score * 100:.0f}%</div>
                        <div class="card-label">Coverage Score</div>
                    </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                    <div class="card">
                        <div class="card-level">{level_icon} {level}</div>
                        <div class="card-label">Niveau détecté</div>
                    </div>""", unsafe_allow_html=True)
        with c3:
            if top_job:
                adequacy_pct = int(top_job['coverage_score'] * 100)
                st.markdown(f"""
                        <div class="card">
                            <div style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--gold);margin-bottom:10px">Meilleur match</div>
                            <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:var(--text);margin-bottom:6px">{top_job['job_title']}</div>
                            <div style="font-size:0.72rem;color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px">Adéquation</div>
                            <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;margin-bottom:6px">
                                <div style="width:{adequacy_pct}%;height:100%;background:linear-gradient(90deg,#c9a84c,#e8c87a);border-radius:2px"></div>
                            </div>
                            <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:var(--gold)">{adequacy_pct}%</div>
                        </div>""", unsafe_allow_html=True)

        # ── Radar ──
        df_blocks = pd.DataFrame([
            {"Bloc": b["block_name"], "Score moyen": b["average_score"], "Score max": b["max_score"]}
            for b in block_scores
        ])
        labels = df_blocks["Bloc"].tolist()
        values = df_blocks["Score moyen"].tolist()

        st.markdown('<div class="label">Cartographie des compétences</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill='toself',
                fillcolor='rgba(201,168,76,0.1)',
                line=dict(color='#c9a84c', width=2)
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor='rgba(255,255,255,0.05)',
                                   color='rgba(255,255,255,0.2)'),
                    angularaxis=dict(color='rgba(255,255,255,0.3)',
                                     gridcolor='rgba(255,255,255,0.05)')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(t=30, b=30, l=30, r=30),
                height=340
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with c2:
            # Progress bars HTML
            bars_html = ""
            for _, row in df_blocks.iterrows():
                pct = int(row["Score moyen"] * 100)
                bars_html += f"""
                <div class="prog-wrap">
                    <div class="prog-header">
                        <span>{row['Bloc']}</span>
                        <span>{row['Score moyen']:.2f}</span>
                    </div>
                    <div class="prog-track">
                        <div class="prog-fill" style="width:{pct}%"></div>
                    </div>
                </div>"""
            st.markdown(f'<div class="card" style="padding-top:32px">{bars_html}</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Heatmap ──
        st.markdown('<div class="label">Heatmap</div>', unsafe_allow_html=True)
        fig_heatmap = px.imshow(
            [df_blocks["Score moyen"]],
            x=df_blocks["Bloc"],
            color_continuous_scale=[[0, "#111118"], [0.5, "#7a5c1e"], [1, "#c9a84c"]],
            aspect="auto"
        )
        fig_heatmap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=10, b=90, l=10, r=80),
            height=160,  # ← plus de hauteur
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                thickness=10,
                len=0.8,
                tickfont=dict(color='rgba(255,255,255,0.4)', size=10),
                tickcolor='rgba(255,255,255,0.2)',
                outlinewidth=0,
            ),
            xaxis=dict(
                color='rgba(255,255,255,0.5)',
                tickangle=-30,
                tickfont=dict(size=11),
            ),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # ── Jobs ──
        st.markdown('<div class="label">Métiers recommandés</div>', unsafe_allow_html=True)
        df_jobs = pd.DataFrame(job_scores[:5])
        fig_jobs = px.bar(
            df_jobs, x="coverage_score", y="job_title", orientation="h",
            color="coverage_score",
            color_continuous_scale=[[0, "#1a1508"], [1, "#c9a84c"]],
        )
        fig_jobs.update_layout(
            yaxis=dict(autorange="reversed", color='rgba(255,255,255,0.5)'),
            xaxis=dict(color='rgba(255,255,255,0.3)', range=[0, 1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10),
            height=240
        )
        fig_jobs.update_traces(marker_line_width=0)
        st.plotly_chart(fig_jobs, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Détail correspondances ──
        st.markdown('<div class="label">Détail des correspondances</div>', unsafe_allow_html=True)
        for block in block_scores[:3]:
            with st.expander(f"{block['block_name']}  ·  {block['average_score']:.2f}"):
                tags_html = "".join(
                    f'<span class="tag">{item["competency"]} {item["score"]:.2f}</span>'
                    for item in block["matched_competencies"]
                )
                st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Explication scores ──
        st.markdown('<div class="label">Explication sémantique</div>', unsafe_allow_html=True)
        for block in block_scores[:3]:
            score = block["average_score"]
            status = ("🟢 Fortement aligné" if score >= 0.65
                      else "🟡 Partiellement couvert" if score >= 0.38
                      else "🔴 À renforcer")
            comps = " · ".join(
                f"{c['competency']} ({c['score']:.2f})"
                for c in block["matched_competencies"][:3]
            )
            st.markdown(f"""
            <div class="card" style="margin-bottom:10px;padding:18px 24px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <span style="font-family:'DM Serif Display',serif;font-size:1rem">{block['block_name']}</span>
                    <span style="font-size:0.8rem;color:var(--muted)">{status}</span>
                </div>
                <div style="font-size:0.8rem;color:var(--muted);line-height:1.6">{comps}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── IA Générative ──
        st.markdown('<div class="label">Analyse IA Générative</div>', unsafe_allow_html=True)
        context, gaps = build_rag_context(user_text, block_scores)
        plan = generate_progression_plan(context)
        bio  = generate_professional_bio(context)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--gold);margin-bottom:10px">Plan de progression</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="plan-box">{plan}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--gold);margin-bottom:10px">Bio professionnelle</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bio-box">{bio}</div>', unsafe_allow_html=True)