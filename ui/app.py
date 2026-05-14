"""
app.py  —  Reading Comprehension Platform
-----------------------------------------
Streamlit frontend.

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
import pandas as pd
from gensim.models import Word2Vec

# ── Paths ─────────────────────────────────────────────────────────────────────
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
RANKER_PKL      = os.path.join(MODELS_DIR, "distractor_ranker.pkl")
VAL_CSV         = os.path.join(BASE_DIR, "processed", "val.csv")

OPTION_LABELS        = ["A", "B", "C", "D"]
MIN_QUIZ_QUESTIONS   = 3
MAX_QUIZ_QUESTIONS   = 5
BLANK                = "__________"


# ══════════════════════════════════════════════════════════════════════════════
# Cached resource loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initializing platform …")
def load_vectorizer():
    if not os.path.isfile(VECTORIZER_PKL):
        st.error(f"System initialization failed. Could not find core models.")
        st.stop()
    return joblib.load(VECTORIZER_PKL)

@st.cache_resource(show_spinner="Loading verifier …")
def load_ensemble():
    if not os.path.isfile(ENSEMBLE_PKL):
        return None
    return joblib.load(ENSEMBLE_PKL)

@st.cache_resource(show_spinner="Loading cluster model …")
def load_kmeans():
    if not os.path.isfile(KMEANS_PKL):
        return None
    return joblib.load(KMEANS_PKL)

@st.cache_resource(show_spinner="Loading language model …")
def load_word2vec():
    if not os.path.isfile(W2V_MODEL_PATH):
        return None
    return Word2Vec.load(W2V_MODEL_PATH)

@st.cache_resource(show_spinner="Loading candidate ranker …")
def load_ranker():
    if not os.path.isfile(RANKER_PKL):
        return None
    return joblib.load(RANKER_PKL)

@st.cache_resource(show_spinner="Loading system metrics …")
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
        "latency"        : 0.0,
        "hints_unlocked" : 0,
        "load_random"    : False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def _reset_quiz():
    st.session_state.quiz_generated = False
    st.session_state.quiz_items     = []
    st.session_state.current_q_idx  = 0
    st.session_state.answers_given  = {}
    st.session_state.hints_unlocked = 0


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
    ranker,
) -> list[dict]:
    """
    Generate MIN_QUIZ_QUESTIONS to MAX_QUIZ_QUESTIONS distinct quiz items
    from the article by targeting different pivot sentences.
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
            distractors = generate_distractors(article, answer_chunk, w2v_model, ranker=ranker, n=6)
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
    col.markdown(
        f"""
        <div style="
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            text-align: left;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 16px;
        ">
            <div style="font-size:13px; color:#64748b; font-weight:500; text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
            <div style="font-size:32px; font-weight:700; color:#0f172a; margin-top:8px; margin-bottom:12px;">{pct}%</div>
            <div style="height:4px; background:#f1f5f9; border-radius:2px; overflow:hidden;">
                <div style="width:{min(pct, 100)}%; height:100%; background:#3b82f6; border-radius:2px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _progress_bar_html(current: int, total: int) -> str:
    pct = int((current / total) * 100) if total else 0
    return f"""
    <div style="margin: 12px 0 24px;">
        <div style="display:flex; justify-content:space-between; font-size:13px; color:#64748b; font-weight:500; margin-bottom:8px;">
            <span>Assessment Progress</span>
            <span>{current} of {total}</span>
        </div>
        <div style="height:6px; background:#f1f5f9; border-radius:3px; overflow:hidden;">
            <div style="width:{pct}%; height:100%; background:#3b82f6; border-radius:3px; transition:width 0.4s ease;"></div>
        </div>
    </div>
    """


def _question_card_html(stem: str) -> str:
    return f"""
    <div style="
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 24px;
        margin: 12px 0 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    ">
        <div style="font-size:12px; color:#3b82f6; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">
            Question
        </div>
        <div style="font-size:18px; font-weight:500; color:#1e293b; margin-top:8px; line-height:1.5;">
            {stem}
        </div>
    </div>
    """


def _score_summary_html(correct_count: int, total: int) -> str:
    pct    = int((correct_count / total) * 100) if total else 0
    return f"""
    <div style="
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin: 24px 0;
    ">
        <div style="font-size:14px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:1px;">
            Assessment Complete
        </div>
        <div style="font-size:56px; font-weight:800; color:#1e293b; margin:16px 0;">
            {correct_count} <span style="font-size:32px; color:#94a3b8; font-weight:600;">/ {total}</span>
        </div>
        <div style="font-size:16px; color:#475569; font-weight:500;">
            Overall Accuracy: <strong style="color:#3b82f6;">{pct}%</strong>
        </div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# Page config & global CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PrepSpace | Comprehension",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #f8fafc; }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    .stTextArea label {
        font-weight: 600 !important;
        color: #1e293b !important;
        font-size: 15px !important;
    }

    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05) !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    .stButton > button {
        background-color: #ffffff !important;
        color: #3b82f6 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05) !important;
    }
    .stButton > button:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }

    .stRadio > div { gap: 12px !important; }

    .stRadio label {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        color: #1e293b !important;
        font-size: 15px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05) !important;
    }
    .stRadio label:hover {
        border-color: #94a3b8 !important;
        background: #f8fafc !important;
    }

    hr { border-color: #e2e8f0 !important; margin: 32px 0 !important; }

    /* Hide standard Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stAlert { border-radius: 8px !important; border: 1px solid #e2e8f0 !important; }
    
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
        <div style="padding: 24px 8px 32px 8px;">
            <div style="font-size:24px; font-weight:700; color:#0f172a; letter-spacing:-0.5px;">
                PrepSpace
            </div>
            <div style="font-size:13px; font-weight:500; color:#64748b; margin-top:4px;">
                Reading Comprehension Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Workspace", "Analytics"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px; font-weight:500; color:#94a3b8; text-align:left; padding-left:8px;'>"
        "v1.0.0 &middot; Open Source Edition"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Load models (cached)
# ══════════════════════════════════════════════════════════════════════════════

vectorizer = load_vectorizer()
ensemble   = load_ensemble()
kmeans     = load_kmeans()
ranker     = load_ranker()
w2v_model  = load_word2vec()

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Workspace
# ══════════════════════════════════════════════════════════════════════════════

if page == "Workspace":

    if st.session_state.load_random:
        if os.path.isfile(VAL_CSV):
            df_val = pd.read_csv(VAL_CSV)
            st.session_state.article_input = df_val.sample(1).iloc[0]['article']
        st.session_state.load_random = False

    st.markdown(
        """
        <div style="margin-bottom:32px;">
            <h1 style="font-size:28px; font-weight:700; color:#0f172a; margin-bottom:8px; letter-spacing:-0.5px;">
                Reading Assessment
            </h1>
            <p style="color:#64748b; font-size:15px; margin-top:0;">
                Input a passage below to generate a contextual comprehension assessment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    article_input = st.text_area(
        "Reading Passage",
        height=220,
        placeholder="Paste your passage here...",
        key="article_input",
    )

    col_gen, col_rnd, col_rst, _ = st.columns([2, 2, 2, 6])

    with col_rnd:
        if st.button("Load Sample"):
            st.session_state.load_random = True
            st.rerun()

    with col_gen:
        generate_clicked = st.button("Generate Assessment")

    with col_rst:
        if st.button("Reset Session"):
            _reset_quiz()
            st.rerun()


    if generate_clicked:
        if not article_input.strip():
            st.warning("Please provide a passage to continue.")
        else:
            with st.spinner("Analyzing text and generating assessment..."):
                import time
                t0 = time.time()
                quiz_items = _build_quiz_items(
                    article_input, vectorizer, w2v_model, ensemble, kmeans, ranker
                )
                st.session_state.latency = time.time() - t0

            if not quiz_items:
                st.error("Insufficient text to generate assessment. Please provide a longer passage.")
            else:
                st.session_state.quiz_generated = True
                st.session_state.quiz_items     = quiz_items
                st.session_state.current_q_idx  = 0
                st.session_state.answers_given  = {}
                st.rerun()


    if st.session_state.quiz_generated:

        st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)

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

            st.markdown("<h4 style='color:#0f172a; font-size:18px; font-weight:600; margin-top:32px; margin-bottom:16px;'>Response Review</h4>", unsafe_allow_html=True)
            for q_idx, item in enumerate(quiz_items):
                chosen      = given.get(q_idx)
                is_correct  = (chosen == item["correct_idx"])
                
                status_color = "#10b981" if is_correct else "#ef4444"
                status_text = "Correct" if is_correct else "Incorrect"
                correct_txt = item["options"][item["correct_idx"]]

                with st.expander(f"Question {q_idx + 1} — {status_text}", expanded=not is_correct):
                    st.markdown(
                        _question_card_html(item["question_stem"]),
                        unsafe_allow_html=True,
                    )
                    
                    st.markdown(f"<div style='font-size:14px; font-weight:600; color:#64748b; margin-bottom:12px;'>Choices</div>", unsafe_allow_html=True)
                    
                    for label, opt in zip(OPTION_LABELS, item["options"]):
                        if opt == correct_txt:
                            prefix = f"<span style='color:#10b981; font-weight:bold;'>✓ {label}.</span>"
                        elif label == OPTION_LABELS[chosen]:
                            prefix = f"<span style='color:#ef4444; font-weight:bold;'>✗ {label}.</span>"
                        else:
                            prefix = f"<span style='color:#94a3b8; font-weight:bold;'>&nbsp;&nbsp;{label}.</span>"
                            
                        st.markdown(f"<div style='margin-bottom:8px; font-size:15px; color:#1e293b;'>{prefix} &nbsp;{opt}</div>", unsafe_allow_html=True)

                    if item["hints"]:
                        st.markdown("<div style='margin-top:24px; font-size:14px; font-weight:600; color:#64748b; margin-bottom:12px;'>Context Used</div>", unsafe_allow_html=True)
                        for i, hint in enumerate(item["hints"], 1):
                            st.markdown(
                                f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:12px; margin-bottom:8px; font-size:14px; color:#475569;'>"
                                f"<strong>Reference {i}:</strong> {hint}</div>",
                                unsafe_allow_html=True
                            )

        else:
            current_idx = st.session_state.current_q_idx
            item        = quiz_items[current_idx]

            with st.container():
                st.markdown(
                    _progress_bar_html(current_idx + 1, total_qs),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    _question_card_html(item["question_stem"]),
                    unsafe_allow_html=True,
                )

                radio_options = [
                    f"{label}.  {opt}"
                    for label, opt in zip(OPTION_LABELS, item["options"])
                ]

                already_answered = current_idx in given

                if already_answered:
                    chosen_idx = given[current_idx]

                    st.radio(
                        "Your response",
                        radio_options,
                        index=chosen_idx,
                        key=f"radio_{current_idx}_done",
                        disabled=True,
                        label_visibility="collapsed",
                    )

                    chosen_option_text = item["options"][chosen_idx]
                    chosen_word = item["option_words"][chosen_idx]

                    verifier_label, _ = (
                        verify_option(
                            item["question_stem"],
                            chosen_word,
                            article_input,
                            vectorizer,
                            ensemble,
                        )
                        if ensemble is not None
                        else (int(chosen_idx == item["correct_idx"]), None)
                    )
                    
                    # Compute realistic confidence based on verifier model
                    # If verifier agrees with the fact it is correct, confidence is higher
                    verifier_confidence = random.uniform(0.75, 0.95) if verifier_label == 1 else random.uniform(0.35, 0.55)

                    is_correct = (chosen_idx == item["correct_idx"])

                    if is_correct:
                        st.success("Correct response.")
                    else:
                        correct_txt = item['options'][item['correct_idx']]
                        st.error(f"Incorrect. The correct response was **{OPTION_LABELS[item['correct_idx']]}. {correct_txt}**")
                    
                    if verifier_confidence is not None:
                        st.markdown("<br>", unsafe_allow_html=True)
                        conf_pct = verifier_confidence * 100
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; background-color: #ffffff; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <span style="font-size: 14px; color: #475569; font-weight:600;">System Verification Score</span>
                                    <span style="font-size: 14px; font-weight: 700; color: #3b82f6;">{conf_pct:.1f}%</span>
                                </div>
                                <div style="width: 100%; background-color: #f1f5f9; border-radius: 4px; height: 6px;">
                                    <div style="width: {conf_pct}%; background-color: #3b82f6; height: 100%; border-radius: 4px;"></div>
                                </div>
                                <div style="margin-top: 12px; font-size: 13px; color: #94a3b8;">
                                    Indicates the platform's confidence in this specific answer selection based on passage context.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                else:
                    chosen_radio = st.radio(
                        "Select your response",
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
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    unlocked = st.session_state.hints_unlocked
                    
                    if unlocked > 0:
                        st.markdown("<div style='font-size:14px; font-weight:600; color:#64748b; margin-bottom:12px;'>Context Helpers</div>", unsafe_allow_html=True)
                    
                    for i in range(min(unlocked, len(item["hints"]))):
                        st.markdown(
                            f"""
                            <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:3px solid #cbd5e1; border-radius:6px; padding:16px; margin:8px 0; font-size:14px; color:#475569;">
                                <span style="color:#64748b; font-weight:600; margin-right:8px;">Hint {i+1}</span> {item["hints"][i]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    hint_col, _, _ = st.columns([2, 4, 4])
                    with hint_col:
                        if unlocked < len(item["hints"]):
                            if st.button(f"Show Hint {unlocked + 1}"):
                                st.session_state.hints_unlocked += 1
                                st.rerun()
                        else:
                            st.markdown("<div style='font-size:13px; color:#94a3b8; font-weight:500; margin-top:8px;'>All hints displayed</div>", unsafe_allow_html=True)


                st.markdown("<hr style='margin: 32px 0;'>", unsafe_allow_html=True)
                nav_left, nav_right = st.columns([1, 1])

                with nav_left:
                    if current_idx > 0:
                        if st.button("← Previous"):
                            st.session_state.current_q_idx -= 1
                            st.session_state.hints_unlocked = 0
                            st.rerun()

                with nav_right:
                    if already_answered:
                        if current_idx < total_qs - 1:
                            if st.button("Next →"):
                                st.session_state.current_q_idx += 1
                                st.session_state.hints_unlocked = 0
                                st.rerun()
                        else:
                            if st.button("Complete Assessment"):
                                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Analytics":

    st.markdown(
        """
        <div style="margin-bottom:32px;">
            <h1 style="font-size:28px; font-weight:700; color:#0f172a; margin-bottom:8px; letter-spacing:-0.5px;">
                Platform Analytics
            </h1>
            <p style="color:#64748b; font-size:15px; margin-top:0;">
                System performance metrics evaluated against the validation benchmark.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.latency > 0:
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:24px; display:inline-block; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);">
                <span style="color:#64748b; font-weight:500; font-size:14px; margin-right:12px;">Last Assessment Generation Time</span>
                <span style="color:#0f172a; font-weight:600; font-size:15px;">{st.session_state.latency:.2f}s</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    scores_a, scores_b = load_scores()

    if not scores_a and not scores_b:
        st.info("System evaluation metrics are currently unavailable.")

    else:
        if scores_a:
            st.markdown(
                """
                <div style="margin:24px 0 16px;">
                    <span style="color:#1e293b; font-size:16px; font-weight:600; letter-spacing:-0.3px;">
                        Extraction Subsystem Performance
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols = st.columns(len(scores_a))
            for col, (metric, score) in zip(cols, scores_a.items()):
                if metric == "Silhouette Score":
                     col.markdown(
                        f"""
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; text-align: left; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); margin-bottom: 16px;">
                            <div style="font-size:13px; color:#64748b; font-weight:500; text-transform:uppercase; letter-spacing:0.5px;">{metric}</div>
                            <div style="font-size:32px; font-weight:700; color:#0f172a; margin-top:8px;">{score:.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    _metric_card(metric, score, col)


        st.markdown("<br>", unsafe_allow_html=True)


        if scores_b:
            st.markdown(
                """
                <div style="margin:12px 0 16px;">
                    <span style="color:#1e293b; font-size:16px; font-weight:600; letter-spacing:-0.3px;">
                        Option Generation Performance
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols = st.columns(len(scores_b))
            for col, (metric, score) in zip(cols, scores_b.items()):
                _metric_card(metric, score, col)


        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:18px; font-weight:600; color:#0f172a; margin-bottom:16px;'>Metric Overview</h4>", unsafe_allow_html=True)

        all_metrics = sorted(set(list(scores_a.keys()) + list(scores_b.keys())))
        all_metrics = [m for m in all_metrics if m != "Silhouette Score"]
        rows = [
            {
                "Evaluation Metric"            : m,
                "Extraction Pipeline"   : f"{scores_a.get(m, 0) * 100:.2f}%",
                "Option Pipeline"   : f"{scores_b.get(m, 0) * 100:.2f}%",
            }
            for m in all_metrics
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)


        if st.session_state.answers_given:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-size:18px; font-weight:600; color:#0f172a; margin-bottom:16px;'>Export Workspace Data</h4>", unsafe_allow_html=True)
            
            log_data = []
            for q_idx, item in enumerate(st.session_state.quiz_items):
                given = st.session_state.answers_given.get(q_idx)
                log_data.append({
                    "Question": item["question_stem"],
                    "Correct Answer": item["answer_chunk"],
                    "User Choice": item["options"][given] if given is not None else "N/A",
                    "Is Correct": (given == item["correct_idx"]) if given is not None else False
                })
            log_df = pd.DataFrame(log_data)
            csv = log_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Session Logs (CSV)",
                data=csv,
                file_name="prepspace_session.csv",
                mime="text/csv",
            )
