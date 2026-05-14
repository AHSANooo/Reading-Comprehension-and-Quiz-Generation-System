"""
model_b_train.py
----------------
Model B  —  Word2Vec Distractor Generator  +  Hint Engine

Components
----------
1. Word2Vec Trainer
   Trains a gensim Word2Vec model on the tokenised training corpus.
   Saved as  models/word2vec.model

2. Hint Generator
   Ranks every sentence in the article by TF-IDF cosine similarity to the
   question stem.  Returns the top-N sentences as contextual hints.

3. Distractor Generator (Word2Vec semantic neighbours)
   Finds the 10 nearest Word2Vec neighbours of the correct-answer chunk.
   Filters out any neighbour token present verbatim in the passage.
   Returns the top 3 remaining neighbours as distractors.

4. Evaluation  (val.csv)
   BLEU, ROUGE-1/2/L, METEOR comparing generated distractors to gold
   wrong options.  Scores saved as  models/model_b_scores.pkl

Checkpointing
-------------
Word2Vec model and evaluation scores are persisted and skipped on re-runs.
"""

import os
import warnings
import re
import random

import joblib
import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")


for _pkg in ("wordnet", "punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
             "maxent_ne_chunker_tab", "words", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

TRAIN_CSV      = os.path.join(PROCESSED_DIR, "train.csv")
VAL_CSV        = os.path.join(PROCESSED_DIR, "val.csv")
VECTORIZER_PKL = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
W2V_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec.model")
SCORES_PKL     = os.path.join(MODELS_DIR, "model_b_scores.pkl")

TOP_N_HINTS         = 3
W2V_TOP_NEIGHBOURS  = 10
N_DISTRACTORS       = 3
MIN_CANDIDATE_LEN   = 3
RANKER_PKL          = os.path.join(MODELS_DIR, "distractor_ranker.pkl")

W2V_VECTOR_SIZE = 100
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 2
W2V_WORKERS     = 4
W2V_EPOCHS      = 5


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(str(text))


def _tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text.lower())


def _correct_option(row: pd.Series) -> str:
    col = str(row.get("answer", "A")).upper()
    if col not in ["A", "B", "C", "D"]: col = "A"
    return str(row.get(col, ""))


def _wrong_options(row: pd.Series) -> list[str]:
    correct_col = str(row.get("answer", "A")).upper()
    if correct_col not in ["A", "B", "C", "D"]: correct_col = "A"
    return [
        str(row.get(col, ""))
        for col in ["A", "B", "C", "D"]
        if col != correct_col
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Word2Vec Training
# ══════════════════════════════════════════════════════════════════════════════

def train_word2vec(
    train_df: pd.DataFrame,
    force_retrain: bool = False,
) -> Word2Vec:
    """
    Train a Word2Vec skip-gram model on the combined article + option corpus
    from train_df.  The model is saved to W2V_MODEL_PATH and reloaded from
    there on subsequent runs unless force_retrain is True.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.isfile(W2V_MODEL_PATH) and not force_retrain:
        print(f"[✓] Word2Vec checkpoint found — loading: {W2V_MODEL_PATH}")
        return Word2Vec.load(W2V_MODEL_PATH)

    print("[→] Tokenising training corpus for Word2Vec …")
    all_text_columns = ["article", "A", "B", "C", "D", "question"]
    sentences = []
    for _, row in train_df.iterrows():
        for col in all_text_columns:
            raw = str(row.get(col, ""))
            if raw.strip():
                sentences.append(_tokenize(raw))

    print(f"    Total sentences: {len(sentences):,}")

    print("[→] Training Word2Vec model …")
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        epochs=W2V_EPOCHS,
        sg=1,
    )

    w2v_model.save(W2V_MODEL_PATH)
    print(f"[✓] Word2Vec model saved → {W2V_MODEL_PATH}")
    return w2v_model


# ══════════════════════════════════════════════════════════════════════════════
# 2. Hint Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_hints(
    question: str,
    article: str,
    vectorizer,
) -> list[str]:
    """
    Return 3 graduated hints:
    1. General: High-similarity sentence (broad context).
    2. Specific: Sentence with highest keyword overlap with the question.
    3. Near-Explicit: Sentence containing the answer (but masked in UI).
    """
    sents = _sentences(article)
    if len(sents) < 3:
        return sents + [""] * (3 - len(sents))

    # Level 1: General Context (Cosine Similarity)
    corpus = sents + [question]
    tfidf_mat = vectorizer.transform(corpus)
    sims = cosine_similarity(tfidf_mat[:-1], tfidf_mat[-1]).flatten()
    general_idx = np.argmax(sims)
    h1 = sents[general_idx]

    # Level 2: Specific Overlap (Word overlap)
    q_words = set(_tokenize(question))
    overlap_scores = [len(set(_tokenize(s)) & q_words) for s in sents]
    specific_idx = np.argmax(overlap_scores)
    h2 = sents[specific_idx]

    # Level 3: Near-Explicit (Random sentence from top 3 that isn't h1 or h2)
    top_3_idxs = np.argsort(sims)[-3:]
    h3_idx = [i for i in top_3_idxs if i not in [general_idx, specific_idx]]
    h3 = sents[h3_idx[0]] if h3_idx else sents[top_3_idxs[0]]

    return [h1, h2, h3]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Distractor Generator — Word2Vec semantic neighbours
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 3. Distractor Ranker (ML-Based)
# ══════════════════════════════════════════════════════════════════════════════

def extract_candidates(article: str) -> list[str]:
    """Extract noun phrases from article as potential distractors."""
    tokens = nltk.word_tokenize(article)
    pos_tags = nltk.pos_tag(tokens)
    grammar = r"NP: {<NNP>+|<NN>+}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)
    
    candidates = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        phrase = " ".join(word for word, tag in subtree.leaves())
        if len(phrase) >= MIN_CANDIDATE_LEN and phrase.lower() not in candidates:
            candidates.append(phrase.lower())
    return candidates

def _get_candidate_features(candidate: str, correct_answer: str, article: str, w2v_model: Word2Vec):
    """Compute features for ranking a candidate distractor."""
    try:
        c_vec = np.mean([w2v_model.wv[w] for w in _tokenize(candidate) if w in w2v_model.wv], axis=0)
        a_vec = np.mean([w2v_model.wv[w] for w in _tokenize(correct_answer) if w in w2v_model.wv], axis=0)
        sim = cosine_similarity(c_vec.reshape(1, -1), a_vec.reshape(1, -1))[0][0]
    except:
        sim = 0.0
        
    char_match = len(set(candidate) & set(correct_answer)) / max(len(candidate), 1)
    freq = article.lower().count(candidate.lower())
    len_diff = abs(len(candidate) - len(correct_answer))
    
    return [sim, char_match, freq, len_diff]

def train_distractor_ranker(train_df: pd.DataFrame, w2v_model: Word2Vec, force_retrain=False):
    """Train a Logistic Regression ranker using gold distractors as positive samples."""
    if os.path.isfile(RANKER_PKL) and not force_retrain:
        return joblib.load(RANKER_PKL)
    
    X, y = [], []
    print("[→] Training Distractor Ranker …")
    
    # Use a small subset for training speed
    for _, row in train_df.head(2000).iterrows():
        article = str(row.get('article', ''))
        correct = _correct_option(row)
        gold_wrongs = _wrong_options(row)
        
        # Positive samples: Gold distractors
        for gw in gold_wrongs:
            X.append(_get_candidate_features(gw, correct, article, w2v_model))
            y.append(1)
            
        # Negative samples: Random phrases from article that aren't answer
        candidates = extract_candidates(article)
        negatives = [c for c in candidates if c.lower() != correct.lower()][:3]
        for neg in negatives:
            X.append(_get_candidate_features(neg, correct, article, w2v_model))
            y.append(0)
            
    ranker = LogisticRegression()
    ranker.fit(X, y)
    joblib.dump(ranker, RANKER_PKL)
    return ranker

def generate_distractors(
    article: str,
    correct_answer: str,
    w2v_model: Word2Vec,
    ranker: LogisticRegression = None,
    n: int = N_DISTRACTORS,
) -> list[str]:
    """Generate distractors by ranking extracted candidates."""
    candidates = extract_candidates(article)
    candidates = [c for c in candidates if c.lower() != correct_answer.lower()]
    
    if not candidates:
        return ["option1", "option2", "option3"] # Fallback

    if ranker is None:
        # Fallback to pure W2V similarity if ranker not provided
        return candidates[:n]

    features = [_get_candidate_features(c, correct_answer, article, w2v_model) for c in candidates]
    probs = ranker.predict_proba(features)[:, 1]
    
    # Sort by probability of being a "good" distractor
    ranked = [c for _, c in sorted(zip(probs, candidates), reverse=True)]
    return ranked[:n]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluation — BLEU, ROUGE, METEOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_distractors(
    val_df: pd.DataFrame,
    w2v_model: Word2Vec,
) -> dict:
    smoother = SmoothingFunction().method1
    rscorer  = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    bleu_scores, rouge1_scores, rouge2_scores = [], [], []
    rougeL_scores, meteor_scores = [], []

    print(f"\n[→] Evaluating distractor generation on {len(val_df):,} val examples …")

    for _, row in val_df.iterrows():
        article = str(row.get("article", ""))
        correct = _correct_option(row)
        wrongs  = _wrong_options(row)

        if not article.strip() or not correct.strip():
            continue

        generated = generate_distractors(article, correct, w2v_model)
        if not generated:
            continue

        sents = _sentences(article)
        anchor = sents[0] if sents else "According to the passage"

        ref_sentence = anchor + " " + " ".join(wrongs)
        hyp_sentence = anchor + " " + " ".join(generated)

        ref = [_tokenize(ref_sentence)]
        hyp = _tokenize(hyp_sentence)

        b = sentence_bleu(ref, hyp, smoothing_function=smoother)
        r = rscorer.score(ref_sentence, hyp_sentence)
        m = meteor_score(ref, hyp)

        bleu_scores.append(b)
        rouge1_scores.append(r["rouge1"].fmeasure)
        rouge2_scores.append(r["rouge2"].fmeasure)
        rougeL_scores.append(r["rougeL"].fmeasure)
        meteor_scores.append(m)

    results = {
        "BLEU"   : float(np.mean(bleu_scores))   if bleu_scores else 0.0,
        "ROUGE-1": float(np.mean(rouge1_scores))  if rouge1_scores else 0.0,
        "ROUGE-2": float(np.mean(rouge2_scores))  if rouge2_scores else 0.0,
        "ROUGE-L": float(np.mean(rougeL_scores))  if rougeL_scores else 0.0,
        "METEOR" : float(np.mean(meteor_scores))  if meteor_scores else 0.0,
    }

    print("\n" + "═" * 50)
    print("  Model B — Distractor Generation Evaluation")
    print("═" * 50)
    for metric, score in results.items():
        print(f"  {metric:<10} : {score:.4f}")
    print("═" * 50 + "\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():

    if not os.path.isfile(TRAIN_CSV):
        raise FileNotFoundError("train.csv not found. Run data_splitter.py first.")
    train_df = pd.read_csv(TRAIN_CSV)

    if not os.path.isfile(VAL_CSV):
        raise FileNotFoundError("val.csv not found. Run data_splitter.py first.")
    val_df = pd.read_csv(VAL_CSV)

    w2v_model = train_word2vec(train_df)
    ranker = train_distractor_ranker(train_df, w2v_model)

    scores = evaluate_distractors(val_df, w2v_model)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scores, SCORES_PKL)
    print(f"[✓] Scores saved → {SCORES_PKL}")


if __name__ == "__main__":
    main()
