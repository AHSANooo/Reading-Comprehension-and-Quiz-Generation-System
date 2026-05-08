"""
app.py  —  AI Reading Comprehension & Quiz Generation System
-------------------------------------------------------------
Streamlit frontend for the Classical-ML quiz pipeline.

Features
--------
• Article input text area
• Quiz view with extracted question + 4 answer options
• Collapsible hint panel
• Analytics dashboard (BLEU, ROUGE, METEOR gauge cards)
• @st.cache_resource for all .pkl artefacts
• st.session_state for full quiz-flow state management

Run locally
-----------
    streamlit run ui/app.py

In Colab / Drive, adjust BASE_DIR below to match your mount path.
"""

import os
import sys

import joblib
import nltk
import streamlit as st

# ── Path resolution: ensure src/ is importable ───────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_SRC    = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model_a_train import extract_question, predict_cluster
from model_b_train import generate_hints, generate_distractors

# ── NLTK bootstrap (silent) ───────────────────────────────────────────────────
for _pkg in ("wordnet", "punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
             "maxent_ne_chunker_tab", "words", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

# ── Configuration & Paths ─────────────────────────────────────────────────────
BASE_DIR   = "/content/drive/MyDrive/AI_Project_2026"
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PKL  = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
VERIFIER_PKL    = os.path.join(MODELS_DIR, "verifier_model.pkl")
KMEANS_PKL      = os.path.join(MODELS_DIR, "kmeans_model.pkl")
SCORES_A_PKL    = os.path.join(MODELS_DIR, "model_a_scores.pkl")
SCORES_B_PKL    = os.path.join(MODELS_DIR, "model_b_scores.pkl")

OPTION_LABELS = ["A", "B", "C", "D"]


# ══════════════════════════════════════════════════════════════════════════════
# Cached resource loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading TF-IDF vectorizer …")
def load_vectorizer():
    if not os.path.isfile(VECTORIZER_PKL):
        st.error(f"Vectorizer not found at `{VECTORIZER_PKL}`. Run `preprocessing.py` first.")
        st.stop()
    return joblib.load(VECTORIZER_PKL)


@st.cache_resource(show_spinner="Loading verifier model …")
def load_verifier():
    if not os.path.isfile(VERIFIER_PKL):
        return None
    return joblib.load(VERIFIER_PKL)


@st.cache_resource(show_spinner="Loading K-Means model …")
def load_kmeans():
    if not os.path.isfile(KMEANS_PKL):
        return None
    return joblib.load(KMEANS_PKL)


@st.cache_resource(show_spinner="Loading evaluation scores …")
def load_scores():
    scores_a = joblib.load(SCORES_A_PKL) if os.path.isfile(SCORES_A_PKL) else {}
    scores_b = joblib.load(SCORES_B_PKL) if os.path.isfile(SCORES_B_PKL) else {}
    return scores_a, scores_b


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "quiz_generated" : False,
        "question"       : "",
        "options"        : [],
        "correct_idx"    : None,
        "selected_idx"   : None,
        "hints"          : [],
        "cluster_id"     : None,
        "answered"       : False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _metric_card(label: str, value: float, col):
    """Render a styled metric card inside a Streamlit column."""
    pct      = round(value * 100, 1)
    colour   = "#4ade80" if pct >= 30 else "#facc15" if pct >= 15 else "#f87171"
    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid {colour}44;
            border-radius: 16px;
            padding: 20px 18px;
            text-align: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        ">
            <div style="font-size:13px; color:#94a3b8; letter-spacing:1px; text-transform:uppercase;">
                {label}
            </div>
            <div style="font-size:38px; font-weight:800; color:{colour}; margin:8px 0 4px;">
                {pct}%
            </div>
            <div style="height:6px; background:#1e293b; border-radius:4px; overflow:hidden;">
                <div style="width:{min(pct, 100)}%; height:100%;
                            background:linear-gradient(90deg,{colour}88,{colour});
                            border-radius:4px;">
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _option_button(label: str, text: str, idx: int):
    """Render a single answer option button and handle selection."""
    answered = st.session_state.answered
    selected = st.session_state.selected_idx
    correct  = st.session_state.correct_idx

    if answered:
        if idx == correct:
            border = "2px solid #4ade80"
            bg     = "#14532d44"
        elif idx == selected and idx != correct:
            border = "2px solid #f87171"
            bg     = "#7f1d1d44"
        else:
            border = "1px solid #334155"
            bg     = "transparent"
    else:
        border = "1px solid #334155"
        bg     = "transparent"

    st.markdown(
        f"""
        <div style="
            background:{bg}; border:{border}; border-radius:12px;
            padding:12px 18px; margin:6px 0; cursor:pointer;
            font-size:15px; color:#e2e8f0;
        ">
            <span style="font-weight:700; color:#818cf8; margin-right:10px;">{label}</span>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not answered:
        if st.button(f"Select {label}", key=f"opt_{idx}", use_container_width=True):
            st.session_state.selected_idx = idx
            st.session_state.answered     = True
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page config & global CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Quiz Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark canvas */
    .stApp { background: #080d16; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
        border-right: 1px solid #1e3a5f44;
    }

    /* Text area */
    .stTextArea textarea {
        background: #111827 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 12px !important;
        font-size: 14px !important;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Option select buttons — make them look minimal */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        color: #475569 !important;
        font-size: 11px !important;
        padding: 2px 8px !important;
    }

    /* Info/warning boxes */
    .stAlert { border-radius: 12px !important; }

    /* Divider */
    hr { border-color: #1e3a5f55 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Navigation
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:20px 0 10px;">
            <div style="font-size:48px;">🧠</div>
            <div style="font-size:20px; font-weight:800;
                        background:linear-gradient(90deg,#818cf8,#c084fc);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                AI Quiz Generator
            </div>
            <div style="font-size:12px; color:#475569; margin-top:4px;">
                Classical ML · RACE Dataset
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📝 Quiz Studio", "📊 Analytics Dashboard"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#334155; text-align:center;'>"
        "TF-IDF · Logistic Regression<br>K-Means · Cosine Similarity"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Load models (cached)
# ══════════════════════════════════════════════════════════════════════════════

vectorizer = load_vectorizer()
verifier   = load_verifier()
kmeans     = load_kmeans()

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Quiz Studio
# ══════════════════════════════════════════════════════════════════════════════

if page == "📝 Quiz Studio":

    st.markdown(
        """
        <h1 style="font-size:32px; font-weight:800;
                   background:linear-gradient(90deg,#818cf8,#c084fc,#38bdf8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   margin-bottom:4px;">
            Reading Comprehension Quiz
        </h1>
        <p style="color:#64748b; font-size:14px; margin-top:0;">
            Paste an article below and let the classical-ML pipeline extract
            a question, distractors, and contextual hints.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Article Input ─────────────────────────────────────────────────────────
    article_input = st.text_area(
        "📄 Article",
        height=260,
        placeholder="Paste your reading passage here …",
        key="article_input",
    )

    col_gen, col_rst, _ = st.columns([2, 2, 8])

    with col_gen:
        generate_clicked = st.button("⚡ Generate Quiz", use_container_width=True)

    with col_rst:
        if st.button("🔄 Reset", use_container_width=True):
            for k in ["quiz_generated", "question", "options", "correct_idx",
                      "selected_idx", "hints", "cluster_id", "answered"]:
                st.session_state[k] = (
                    False if k in ("quiz_generated", "answered") else
                    None  if k in ("correct_idx", "selected_idx", "cluster_id") else
                    []    if k in ("options", "hints") else
                    ""
                )
            st.rerun()

    # ── Generate logic ────────────────────────────────────────────────────────
    if generate_clicked:
        if not article_input.strip():
            st.warning("Please paste an article first.")
        else:
            with st.spinner("Extracting question and generating options …"):
                # Step 1: pick a "correct answer" as the most informative sentence
                sents       = nltk.sent_tokenize(article_input)
                pivot       = sents[len(sents) // 2] if sents else article_input[:200]
                question    = extract_question(article_input, pivot, vectorizer)
                distractors = generate_distractors(article_input, pivot, vectorizer)
                hints_list  = generate_hints(question, article_input, vectorizer)

                # Build 4 options: correct answer (pivot) + 3 distractors
                import random
                options_raw = [pivot] + distractors[:3]
                while len(options_raw) < 4:
                    options_raw.append("(No further distractor found)")

                correct_idx = 0
                combined    = list(enumerate(options_raw))
                random.shuffle(combined)
                shuffled_options = [t for _, t in combined]
                new_correct_idx  = next(
                    i for i, (orig, _) in enumerate(combined) if orig == correct_idx
                )

                # Verifier confidence (if model available)
                if verifier is not None:
                    import numpy as np
                    vecs  = vectorizer.transform(
                        [article_input + " " + opt for opt in shuffled_options]
                    )
                    probs = verifier.predict_proba(vecs)[:, 1]
                    new_correct_idx = int(np.argmax(probs))

                # Cluster
                cluster_id = None
                if kmeans is not None:
                    cluster_id = predict_cluster(question, pivot, vectorizer, kmeans)

                st.session_state.quiz_generated = True
                st.session_state.question       = question
                st.session_state.options        = shuffled_options
                st.session_state.correct_idx    = new_correct_idx
                st.session_state.selected_idx   = None
                st.session_state.hints          = hints_list
                st.session_state.cluster_id     = cluster_id
                st.session_state.answered       = False
            st.rerun()

    # ── Quiz view ─────────────────────────────────────────────────────────────
    if st.session_state.quiz_generated:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Cluster badge
        if st.session_state.cluster_id is not None:
            st.markdown(
                f"<span style='background:#312e81; color:#a5b4fc; padding:4px 12px;"
                f"border-radius:20px; font-size:12px; font-weight:600;'>"
                f"Cluster #{st.session_state.cluster_id}</span>",
                unsafe_allow_html=True,
            )

        # Question
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg,#1e1b4b,#0f172a);
                border-left: 4px solid #818cf8;
                border-radius: 12px;
                padding: 20px 24px;
                margin: 16px 0;
            ">
                <div style="font-size:12px; color:#818cf8; font-weight:600;
                            letter-spacing:1px; text-transform:uppercase;">
                    Question
                </div>
                <div style="font-size:18px; font-weight:600; color:#e2e8f0; margin-top:8px;">
                    {st.session_state.question}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Options
        st.markdown(
            "<div style='font-size:12px; color:#64748b; font-weight:600; "
            "letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>"
            "Choose an Answer</div>",
            unsafe_allow_html=True,
        )
        for idx, (label, option) in enumerate(
            zip(OPTION_LABELS, st.session_state.options)
        ):
            _option_button(label, option, idx)

        # Feedback after answering
        if st.session_state.answered:
            if st.session_state.selected_idx == st.session_state.correct_idx:
                st.success("✅ Correct! Well done.")
            else:
                correct_label = OPTION_LABELS[st.session_state.correct_idx]
                st.error(
                    f"❌ Incorrect. The correct answer was **{correct_label}**: "
                    f"{st.session_state.options[st.session_state.correct_idx]}"
                )

        # ── Hint panel (collapsible) ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("💡 Contextual Hints", expanded=False):
            if st.session_state.hints:
                for i, hint in enumerate(st.session_state.hints, 1):
                    st.markdown(
                        f"""
                        <div style="
                            background:#0f2744; border:1px solid #1e3a5f;
                            border-radius:10px; padding:12px 16px; margin:8px 0;
                            font-size:14px; color:#cbd5e1; line-height:1.6;
                        ">
                            <span style="color:#38bdf8; font-weight:700;">
                                Hint {i}&nbsp;
                            </span>
                            {hint}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No hints available for this article.")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics Dashboard":

    st.markdown(
        """
        <h1 style="font-size:32px; font-weight:800;
                   background:linear-gradient(90deg,#38bdf8,#818cf8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   margin-bottom:4px;">
            Analytics Dashboard
        </h1>
        <p style="color:#64748b; font-size:14px; margin-top:0;">
            BLEU · ROUGE · METEOR scores computed on the validation split.
        </p>
        """,
        unsafe_allow_html=True,
    )

    scores_a, scores_b = load_scores()

    if not scores_a and not scores_b:
        st.warning(
            "No evaluation scores found. "
            "Run `model_a_train.py` and `model_b_train.py` first, then reload."
        )
    else:
        # ── Model A ───────────────────────────────────────────────────────────
        if scores_a:
            st.markdown(
                """
                <div style="margin:24px 0 12px;">
                    <span style="background:#1e1b4b; color:#818cf8;
                                 padding:4px 14px; border-radius:20px;
                                 font-size:13px; font-weight:700;">
                        Model A — Question Extraction
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols = st.columns(len(scores_a))
            for col, (metric, score) in zip(cols, scores_a.items()):
                _metric_card(metric, score, col)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Model B ───────────────────────────────────────────────────────────
        if scores_b:
            st.markdown(
                """
                <div style="margin:12px 0 12px;">
                    <span style="background:#0c2a1a; color:#4ade80;
                                 padding:4px 14px; border-radius:20px;
                                 font-size:13px; font-weight:700;">
                        Model B — Distractor Generation
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols = st.columns(len(scores_b))
            for col, (metric, score) in zip(cols, scores_b.items()):
                _metric_card(metric, score, col)

        # ── Combined comparison table ─────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Score Comparison")

        import pandas as pd
        rows = []
        all_metrics = sorted(set(list(scores_a.keys()) + list(scores_b.keys())))
        for m in all_metrics:
            rows.append({
                "Metric"       : m,
                "Model A (Q.Ext)" : f"{scores_a.get(m, 0)*100:.2f}%",
                "Model B (Dist.)" : f"{scores_b.get(m, 0)*100:.2f}%",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Methodology note ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Methodology", expanded=False):
            st.markdown(
                """
                | Component | Technique |
                |-----------|-----------|
                | Vectorisation | TF-IDF (`sublinear_tf=True`, bigrams, 50k features) |
                | Question Extraction | Cosine Similarity — best article sentence ↔ correct answer |
                | Answer Verification | Logistic Regression on TF-IDF vectors (`saga` solver) |
                | Clustering | Mini-Batch K-Means (k=10) on question-answer pair vectors |
                | Hint Generation | Top-N article sentences by cosine similarity to question |
                | Distractor Generation | NP-chunk candidates filtered to medium/low similarity band |
                | Evaluation | BLEU (NLTK), ROUGE-1/2/L (`rouge-score`), METEOR (NLTK) |
                """
            )
