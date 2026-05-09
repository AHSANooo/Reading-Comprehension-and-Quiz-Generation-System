"""
app.py  —  AI Reading Comprehension & Quiz Generation System
-------------------------------------------------------------
Streamlit frontend for the Classical-ML quiz pipeline.

Features
--------
• Multi-question quiz  (3-5 fill-in-the-blank questions per article)
• st.radio for answer selection with shuffled options
• Hint panel that masks the correct answer string
• Analytics dashboard with BLEU / ROUGE / METEOR gauge cards
• st.cache_resource for all .pkl and Word2Vec artefacts
• st.session_state for complete quiz-flow state management

Run locally
-----------
    streamlit run ui/app.py
"""

import os
import random
import re
import sys

import joblib
import nltk
import numpy as np
import streamlit as st
from gensim.models import Word2Vec

BASE_DIR = "/content/drive/MyDrive/AI_Project_2026"
if not os.path.isdir(BASE_DIR):
    # Hardcoded local path to prevent stale terminal/Trash issues
    LOCAL_BASE = "/home/ahsan/Documents/Uni work/Sem 6/AI Lab/Project/AI_Project_2026"
    if os.path.isdir(LOCAL_BASE):
        BASE_DIR = LOCAL_BASE
    else:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_SRC = os.path.join(BASE_DIR, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model_a_train import extract_question, predict_cluster, verify_option
from model_b_train import generate_hints, generate_distractors


for _pkg in ("wordnet", "punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
             "maxent_ne_chunker_tab", "words", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


MODELS_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PKL  = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
ENSEMBLE_PKL    = os.path.join(MODELS_DIR, "verifier_ensemble_model.pkl")
KMEANS_PKL      = os.path.join(MODELS_DIR, "kmeans_model.pkl")
W2V_MODEL_PATH  = os.path.join(MODELS_DIR, "word2vec.model")
SCORES_A_PKL    = os.path.join(MODELS_DIR, "model_a_scores.pkl")
SCORES_B_PKL    = os.path.join(MODELS_DIR, "model_b_scores.pkl")

OPTION_LABELS        = ["A", "B", "C", "D"]
MIN_QUIZ_QUESTIONS   = 3
MAX_QUIZ_QUESTIONS   = 5
BLANK                = "__________"


# ══════════════════════════════════════════════════════════════════════════════
# Cached resource loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading TF-IDF vectorizer …")
def load_vectorizer():
    if not os.path.isfile(VECTORIZER_PKL):
        st.error(
            f"Vectorizer not found at `{VECTORIZER_PKL}`. "
            "Run `preprocessing.py` first."
        )
        st.stop()
    return joblib.load(VECTORIZER_PKL)


@st.cache_resource(show_spinner="Loading ensemble verifier …")
def load_ensemble():
    if not os.path.isfile(ENSEMBLE_PKL):
        return None
    return joblib.load(ENSEMBLE_PKL)


@st.cache_resource(show_spinner="Loading K-Means model …")
def load_kmeans():
    if not os.path.isfile(KMEANS_PKL):
        return None
    return joblib.load(KMEANS_PKL)


@st.cache_resource(show_spinner="Loading Word2Vec model …")
def load_word2vec():
    if not os.path.isfile(W2V_MODEL_PATH):
        return None
    return Word2Vec.load(W2V_MODEL_PATH)


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
        "quiz_items"     : [],
        "current_q_idx"  : 0,
        "answers_given"  : {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_quiz():
    st.session_state.quiz_generated = False
    st.session_state.quiz_items     = []
    st.session_state.current_q_idx  = 0
    st.session_state.answers_given  = {}


# ══════════════════════════════════════════════════════════════════════════════
# Quiz generation logic
# ══════════════════════════════════════════════════════════════════════════════

def _mask_answer_in_hint(hint: str, answer_chunk: str) -> str:
    """
    Replace exact occurrences of *answer_chunk* inside *hint* with
    "[answer]" so the hint is useful without giving away the answer.
    """
    if not answer_chunk:
        return hint
    return re.sub(re.escape(answer_chunk), "[answer]", hint, flags=re.IGNORECASE)


def _build_quiz_items(
    article: str,
    vectorizer,
    w2v_model,
    ensemble,
    kmeans,
) -> list[dict]:
    """
    Generate MIN_QUIZ_QUESTIONS to MAX_QUIZ_QUESTIONS distinct quiz items
    from the article by targeting different pivot sentences.

    Each item is a dict with keys:
      question_stem, answer_chunk, options, correct_idx, hints, cluster_id
    """
    all_sentences = nltk.sent_tokenize(article)
    n_questions   = min(
        MAX_QUIZ_QUESTIONS,
        max(MIN_QUIZ_QUESTIONS, len(all_sentences) // 3)
    )

    stride        = max(1, len(all_sentences) // n_questions)
    pivot_indices = [i * stride for i in range(n_questions) if i * stride < len(all_sentences)]

    quiz_items = []

    for pivot_idx in pivot_indices[:n_questions]:
        pivot_sentence = all_sentences[pivot_idx]

        question_stem, answer_chunk = extract_question(
            article, pivot_sentence, vectorizer
        )

        if not answer_chunk or answer_chunk == pivot_sentence:
            continue

        if w2v_model is not None:
            distractors = generate_distractors(article, answer_chunk, w2v_model, n=6)
        else:
            distractors = []

        clean_distractors = []
        for d in distractors:
            if d.lower() != answer_chunk.lower() and d.lower() not in [c.lower() for c in clean_distractors]:
                clean_distractors.append(d)

        is_cap = answer_chunk and answer_chunk[0].isupper()
        fmt_distractors = [d.title() if is_cap else d.lower() for d in clean_distractors[:3]]

        correct_sentence = question_stem.replace("__________", answer_chunk)
        options_pool = [(correct_sentence, True, answer_chunk)]

        for d in fmt_distractors:
            wrong_sentence = question_stem.replace("__________", d)
            options_pool.append((wrong_sentence, False, d))

        while len(options_pool) < 4:
            options_pool.append(("(No distractor available)", False, ""))

        random.shuffle(options_pool)

        shuffled_options = [text for text, _, _ in options_pool]
        shuffled_words = [word for _, _, word in options_pool]
        shuffled_correct_idx = next(
            i for i, (_, is_true, _) in enumerate(options_pool) if is_true
        )

        raw_hints = generate_hints(question_stem, article, vectorizer)
        masked_hints = [_mask_answer_in_hint(h, answer_chunk) for h in raw_hints]

        cluster_id = None
        if kmeans is not None:
            cluster_id = predict_cluster(question_stem, answer_chunk, vectorizer, kmeans)

        quiz_items.append({
            "question_stem": question_stem,
            "answer_chunk": answer_chunk,
            "options": shuffled_options,
            "option_words": shuffled_words,
            "correct_idx": shuffled_correct_idx,
            "hints": masked_hints,
            "cluster_id": cluster_id,
        })

    return quiz_items


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _metric_card(label: str, value: float, col):
    pct    = round(value * 100, 1)
    colour = "#4ade80" if pct >= 30 else "#facc15" if pct >= 15 else "#f87171"
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
            <div style="font-size:13px; color:#94a3b8; letter-spacing:1px;
                        text-transform:uppercase;">{label}</div>
            <div style="font-size:38px; font-weight:800; color:{colour};
                        margin:8px 0 4px;">{pct}%</div>
            <div style="height:6px; background:#1e293b; border-radius:4px; overflow:hidden;">
                <div style="width:{min(pct, 100)}%; height:100%;
                            background:linear-gradient(90deg,{colour}88,{colour});
                            border-radius:4px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _progress_bar_html(current: int, total: int) -> str:
    pct = int((current / total) * 100) if total else 0
    return f"""
    <div style="margin: 12px 0 20px;">
        <div style="display:flex; justify-content:space-between;
                    font-size:12px; color:#64748b; margin-bottom:6px;">
            <span>Question {current} of {total}</span>
            <span>{pct}% complete</span>
        </div>
        <div style="height:6px; background:#1e293b; border-radius:4px; overflow:hidden;">
            <div style="width:{pct}%; height:100%;
                        background:linear-gradient(90deg,#818cf8,#c084fc);
                        border-radius:4px; transition:width 0.4s ease;"></div>
        </div>
    </div>
    """


def _question_card_html(stem: str, cluster_id) -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg,#1e1b4b,#0f172a);
        border-left: 4px solid #818cf8;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0 20px;
    ">
        <div style="font-size:12px; color:#818cf8; font-weight:600; text-transform:uppercase;">
            Fact Verification
        </div>
        <div style="font-size:19px; font-weight:600; color:#e2e8f0; margin-top:10px;">
            Identify the factually correct statement based on the article:
        </div>
    </div>
    """


def _score_summary_html(correct_count: int, total: int) -> str:
    pct    = int((correct_count / total) * 100) if total else 0
    colour = "#4ade80" if pct >= 70 else "#facc15" if pct >= 40 else "#f87171"
    return f"""
    <div style="
        background: linear-gradient(135deg,#0f2744,#0a1628);
        border: 1px solid {colour}55;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
    ">
        <div style="font-size:14px; color:#64748b; text-transform:uppercase;
                    letter-spacing:1px;">Quiz Complete</div>
        <div style="font-size:56px; font-weight:800; color:{colour};
                    margin:12px 0;">{correct_count}/{total}</div>
        <div style="font-size:16px; color:#94a3b8;">
            You scored <strong style="color:{colour};">{pct}%</strong>
        </div>
    </div>
    """


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

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #080d16; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
        border-right: 1px solid #1e3a5f44;
    }

    .stTextArea textarea {
        background: #111827 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 12px !important;
        font-size: 14px !important;
    }

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

    .stRadio > div { gap: 8px !important; }

    .stRadio label {
        background: #111827 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        color: #e2e8f0 !important;
        font-size: 15px !important;
        cursor: pointer !important;
        transition: border-color 0.2s !important;
    }
    .stRadio label:hover { border-color: #818cf8 !important; }

    .stAlert { border-radius: 12px !important; }

    hr { border-color: #1e3a5f55 !important; }

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
        "TF-IDF · Soft Voting Ensemble<br>Word2Vec · K-Means"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Load models (cached)
# ══════════════════════════════════════════════════════════════════════════════

vectorizer = load_vectorizer()
ensemble   = load_ensemble()
kmeans     = load_kmeans()
w2v_model  = load_word2vec()

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
            Paste an article and the pipeline will generate a multi-question
            fill-in-the-blank quiz with Word2Vec distractors and contextual hints.
        </p>
        """,
        unsafe_allow_html=True,
    )

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
            _reset_quiz()
            st.rerun()


    if generate_clicked:
        if not article_input.strip():
            st.warning("Please paste an article first.")
        else:
            with st.spinner("Building your multi-question quiz …"):
                quiz_items = _build_quiz_items(
                    article_input, vectorizer, w2v_model, ensemble, kmeans
                )

            if not quiz_items:
                st.error("Could not extract questions from this article. Try a longer passage.")
            else:
                st.session_state.quiz_generated = True
                st.session_state.quiz_items     = quiz_items
                st.session_state.current_q_idx  = 0
                st.session_state.answers_given  = {}
                st.rerun()


    if st.session_state.quiz_generated:

        quiz_items  = st.session_state.quiz_items
        total_qs    = len(quiz_items)
        given       = st.session_state.answers_given
        all_done    = len(given) == total_qs

        if all_done:
            correct_count = sum(
                1 for q_idx, chosen in given.items()
                if chosen == quiz_items[q_idx]["correct_idx"]
            )
            st.markdown(_score_summary_html(correct_count, total_qs),
                        unsafe_allow_html=True)

            st.markdown("#### Question Review")
            for q_idx, item in enumerate(quiz_items):
                chosen      = given.get(q_idx)
                is_correct  = (chosen == item["correct_idx"])
                icon        = "✅" if is_correct else "❌"
                correct_txt = item["options"][item["correct_idx"]]

                with st.expander(
                    f"{icon}  Q{q_idx + 1}: {item['question_stem'][:80]}…",
                    expanded=not is_correct,
                ):
                    st.markdown(
                        _question_card_html(
                            item["question_stem"], item["cluster_id"]
                        ),
                        unsafe_allow_html=True,
                    )
                    for label, opt in zip(OPTION_LABELS, item["options"]):
                        prefix = "✅" if opt == correct_txt else (
                            "❌" if label == OPTION_LABELS[chosen] else "◦"
                        )
                        st.markdown(f"**{prefix} {label}.** {opt}")

                    if item["hints"]:
                        with st.expander("💡 Hints", expanded=False):
                            for i, hint in enumerate(item["hints"], 1):
                                st.info(f"**Hint {i}:** {hint}")

        else:
            current_idx = st.session_state.current_q_idx
            item        = quiz_items[current_idx]

            with st.container(border=True):
                st.markdown(
                    _progress_bar_html(current_idx + 1, total_qs),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    _question_card_html(item["question_stem"], item["cluster_id"]),
                    unsafe_allow_html=True,
                )

                radio_options = [
                    f"{label}. {opt}"
                    for label, opt in zip(OPTION_LABELS, item["options"])
                ]

                already_answered = current_idx in given

                if already_answered:
                    chosen_idx = given[current_idx]

                    st.radio(
                        "Your answer",
                        radio_options,
                        index=chosen_idx,
                        key=f"radio_{current_idx}_done",
                        disabled=True,
                        label_visibility="collapsed",
                    )

                    chosen_option_text = item["options"][chosen_idx]
                    chosen_word = item["option_words"][chosen_idx]

                    verifier_label, verifier_confidence = (
                        verify_option(
                            chosen_word,
                            "",
                            article_input,
                            vectorizer,
                            ensemble,
                        )
                        if ensemble is not None
                        else (int(chosen_idx == item["correct_idx"]), None)
                    )

                    is_correct = (chosen_idx == item["correct_idx"])

                    if is_correct:
                        st.success(f"✅ Correct!")
                    else:
                        correct_txt = item['options'][item['correct_idx']]
                        st.error(
                            f"❌ Incorrect. Correct answer: "
                            f"**{OPTION_LABELS[item['correct_idx']]}. {correct_txt}**"
                        )
                    
                    if verifier_confidence is not None:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("##### 🤖 ML Verifier Evaluation")
                        conf_pct = verifier_confidence * 100
                        color = "green" if conf_pct >= 70 else "orange" if conf_pct >= 40 else "red"
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #334155; border-radius: 8px; padding: 16px; background-color: #0f172a;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="font-size: 14px; color: #94a3b8;">Confidence Score</span>
                                    <span style="font-size: 14px; font-weight: bold; color: {color};">{conf_pct:.1f}%</span>
                                </div>
                                <div style="width: 100%; background-color: #1e293b; border-radius: 4px; height: 8px;">
                                    <div style="width: {conf_pct}%; background-color: {color}; height: 100%; border-radius: 4px;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                else:
                    chosen_radio = st.radio(
                        "Choose your answer",
                        radio_options,
                        index=None,
                        key=f"radio_{current_idx}",
                        label_visibility="collapsed",
                    )

                    if chosen_radio is not None:
                        chosen_idx = radio_options.index(chosen_radio)
                        st.session_state.answers_given[current_idx] = chosen_idx
                        st.rerun()


                if item["hints"]:
                    with st.expander("💡 Contextual Hints", expanded=False):
                        for i, hint in enumerate(item["hints"], 1):
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


                nav_left, nav_right = st.columns([1, 1])

                with nav_left:
                    if current_idx > 0:
                        if st.button("← Previous", use_container_width=True):
                            st.session_state.current_q_idx -= 1
                            st.rerun()

                with nav_right:
                    if already_answered:
                        if current_idx < total_qs - 1:
                            if st.button("Next →", use_container_width=True):
                                st.session_state.current_q_idx += 1
                                st.rerun()
                        else:
                            if st.button("🏁 Finish Quiz", use_container_width=True):
                                st.rerun()


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


        if scores_b:
            st.markdown(
                """
                <div style="margin:12px 0 12px;">
                    <span style="background:#0c2a1a; color:#4ade80;
                                 padding:4px 14px; border-radius:20px;
                                 font-size:13px; font-weight:700;">
                        Model B — Distractor Generation (Word2Vec)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols = st.columns(len(scores_b))
            for col, (metric, score) in zip(cols, scores_b.items()):
                _metric_card(metric, score, col)


        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Score Comparison")

        import pandas as pd

        all_metrics = sorted(set(list(scores_a.keys()) + list(scores_b.keys())))
        rows = [
            {
                "Metric"            : m,
                "Model A (Q.Ext)"   : f"{scores_a.get(m, 0) * 100:.2f}%",
                "Model B (Dist.)"   : f"{scores_b.get(m, 0) * 100:.2f}%",
            }
            for m in all_metrics
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Methodology", expanded=False):
            st.markdown(
                """
                | Component | Technique |
                |-----------|-----------|
                | Vectorisation | TF-IDF (`sublinear_tf=True`, bigrams, 50k features) |
                | Question Extraction | Cosine Similarity → fill-in-the-blank (POS noun chunk) |
                | Answer Verification | Soft Voting Ensemble (LR + MNB + SGD) |
                | Clustering | Mini-Batch K-Means (k=10) on Q-A pair vectors |
                | Hint Generation | Top-N article sentences by cosine similarity to question |
                | Distractor Generation | Word2Vec semantic neighbours, passage-filtered |
                | Evaluation | BLEU (NLTK), ROUGE-1/2/L (`rouge-score`), METEOR (NLTK) |
                """
            )
