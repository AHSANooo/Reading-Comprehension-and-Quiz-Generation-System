"""
model_a_train.py
----------------
Model A  —  Question Extractor  +  Answer Verifier  +  K-Means Clusterer

Components
----------
1. Question Extractor
   Scores every sentence in the article against the correct answer using
   cosine similarity (TF-IDF vectors).  The highest-scoring sentence is
   returned as the "generated question stem."

2. Verifier (Logistic Regression)
   Binary classifier: given a TF-IDF vector of (article + option), predict
   whether that option is the correct answer (label = 1) or not (label = 0).
   Saved as  models/verifier_model.pkl

3. Clusterer (K-Means)
   Clusters question-answer pair vectors into K groups for thematic
   organisation.
   Saved as  models/kmeans_model.pkl

4. Evaluation (Question Extraction quality on val.csv)
   Compares extracted sentence to the gold-standard question using
   BLEU, ROUGE-1/2/L, and METEOR.  Results are printed to stdout.

Checkpointing
-------------
Every heavy artefact is saved as a .pkl.  If the file exists on the next
run the computation is skipped entirely.
"""

import os
import re
import warnings

import joblib
import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Download required NLTK data once ─────────────────────────────────────────
for _pkg in ("wordnet", "punkt", "punkt_tab", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = "/content/drive/MyDrive/AI_Project_2026"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

TRAIN_CSV        = os.path.join(PROCESSED_DIR, "train.csv")
VAL_CSV          = os.path.join(PROCESSED_DIR, "val.csv")
VECTORIZER_PKL   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
TRAIN_MATRIX_PKL = os.path.join(MODELS_DIR, "tfidf_train_matrix.pkl")
VERIFIER_PKL     = os.path.join(MODELS_DIR, "verifier_model.pkl")
KMEANS_PKL       = os.path.join(MODELS_DIR, "kmeans_model.pkl")

# K-Means hyper-parameters
N_CLUSTERS   = 10
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK's Punkt tokeniser."""
    return nltk.sent_tokenize(str(text))


def _correct_option(row: pd.Series) -> str:
    """Return the text of the correct answer for a given row."""
    mapping = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
    col = mapping.get(str(row.get("answer", "A")).upper(), "opa")
    return str(row.get(col, ""))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Question Extraction via Cosine Similarity
# ══════════════════════════════════════════════════════════════════════════════

def extract_question(
    article: str,
    correct_answer: str,
    vectorizer,
) -> str:
    """
    Return the sentence from *article* that is most similar to
    *correct_answer* in TF-IDF cosine-similarity space.
    """
    sents = _sentences(article)
    if not sents:
        return ""

    # Transform sentences + answer into sparse vectors
    corpus    = sents + [correct_answer]
    tfidf_mat = vectorizer.transform(corpus)           # sparse (n+1, vocab)

    sent_vecs  = tfidf_mat[:-1]                        # (n, vocab)
    answer_vec = tfidf_mat[-1]                         # (1, vocab)

    sims = cosine_similarity(sent_vecs, answer_vec).flatten()
    best_idx = int(np.argmax(sims))
    return sents[best_idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Verifier — Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════

def _build_verifier_dataset(
    df: pd.DataFrame, vectorizer
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    For each row create 4 (article+option, label) pairs:
      label=1  if option matches the correct answer
      label=0  otherwise
    Returns (X_sparse, y).
    """
    texts, labels = [], []
    option_cols   = ["opa", "opb", "opc", "opd"]
    answer_map    = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}

    for _, row in df.iterrows():
        article = str(row.get("article", ""))
        correct = answer_map.get(str(row.get("answer", "A")).upper(), "opa")
        for col in option_cols:
            option = str(row.get(col, ""))
            texts.append(article + " " + option)
            labels.append(1 if col == correct else 0)

    X = vectorizer.transform(texts)          # scipy sparse
    y = np.array(labels, dtype=np.int8)
    return X, y


def train_verifier(
    train_df: pd.DataFrame,
    train_matrix: sp.csr_matrix,
    vectorizer,
    force_retrain: bool = False,
) -> LogisticRegression:
    """
    Train (or load from checkpoint) the Logistic Regression verifier.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.isfile(VERIFIER_PKL) and not force_retrain:
        print(f"[✓] Verifier checkpoint found — loading: {VERIFIER_PKL}")
        return joblib.load(VERIFIER_PKL)

    print("[→] Building verifier training dataset …")
    X_train, y_train = _build_verifier_dataset(train_df, vectorizer)
    print(f"    Samples : {X_train.shape[0]:,}  |  Features : {X_train.shape[1]:,}")
    print(f"    Positive: {y_train.sum():,}  |  Negative: {(y_train == 0).sum():,}")

    print("[→] Training Logistic Regression verifier …")
    clf = LogisticRegression(
        max_iter=1_000,
        C=1.0,
        solver="saga",          # handles sparse input efficiently
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, VERIFIER_PKL)
    print(f"[✓] Verifier saved → {VERIFIER_PKL}")
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# 3. Clusterer — K-Means
# ══════════════════════════════════════════════════════════════════════════════

def _build_cluster_matrix(
    df: pd.DataFrame, vectorizer
) -> sp.csr_matrix:
    """Vectorise (question + correct_answer) strings for clustering."""
    texts = []
    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        answer   = _correct_option(row)
        texts.append(question + " " + answer)
    return vectorizer.transform(texts)


def train_kmeans(
    train_df: pd.DataFrame,
    vectorizer,
    force_retrain: bool = False,
) -> MiniBatchKMeans:
    """
    Cluster question-answer pairs with Mini-Batch K-Means (sparse-compatible).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.isfile(KMEANS_PKL) and not force_retrain:
        print(f"[✓] K-Means checkpoint found — loading: {KMEANS_PKL}")
        return joblib.load(KMEANS_PKL)

    print("[→] Building clustering matrix …")
    X_cluster = _build_cluster_matrix(train_df, vectorizer)
    print(f"    Matrix shape: {X_cluster.shape}")

    print(f"[→] Fitting Mini-Batch K-Means (k={N_CLUSTERS}) …")
    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=1_024,
        n_init=10,
    )
    km.fit(X_cluster)
    print(f"    Inertia: {km.inertia_:.2f}")

    joblib.dump(km, KMEANS_PKL)
    print(f"[✓] K-Means model saved → {KMEANS_PKL}")
    return km


def predict_cluster(
    question: str,
    answer: str,
    vectorizer,
    kmeans: MiniBatchKMeans,
) -> int:
    """Return the cluster ID for a given question-answer pair."""
    vec = vectorizer.transform([question + " " + answer])
    return int(kmeans.predict(vec)[0])


# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluation — BLEU, ROUGE, METEOR
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text.lower())


def evaluate_extraction(val_df: pd.DataFrame, vectorizer) -> dict:
    """
    For every row in val_df:
      - extract the best sentence using cosine similarity
      - compare it to the gold-standard question field
    Aggregate BLEU, ROUGE-1/2/L, and METEOR; print and return scores.
    """
    smoother = SmoothingFunction().method1
    rscorer  = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    bleu_scores   = []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    meteor_scores = []

    print(f"\n[→] Evaluating question extraction on {len(val_df):,} val examples …")

    for _, row in val_df.iterrows():
        article = str(row.get("article", ""))
        gold_q  = str(row.get("question", ""))
        answer  = _correct_option(row)

        if not article.strip() or not gold_q.strip():
            continue

        extracted = extract_question(article, answer, vectorizer)

        ref  = [_tokenize(gold_q)]
        hyp  = _tokenize(extracted)

        # BLEU
        bleu_scores.append(sentence_bleu(ref, hyp, smoothing_function=smoother))

        # ROUGE
        r = rscorer.score(gold_q, extracted)
        rouge1_scores.append(r["rouge1"].fmeasure)
        rouge2_scores.append(r["rouge2"].fmeasure)
        rougeL_scores.append(r["rougeL"].fmeasure)

        # METEOR  (reference must be tokenised list)
        meteor_scores.append(meteor_score(ref, hyp))

    results = {
        "BLEU"   : float(np.mean(bleu_scores)),
        "ROUGE-1": float(np.mean(rouge1_scores)),
        "ROUGE-2": float(np.mean(rouge2_scores)),
        "ROUGE-L": float(np.mean(rougeL_scores)),
        "METEOR" : float(np.mean(meteor_scores)),
    }

    print("\n" + "═" * 50)
    print("  Model A — Question Extraction Evaluation")
    print("═" * 50)
    for metric, score in results.items():
        print(f"  {metric:<10} : {score:.4f}")
    print("═" * 50 + "\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load artefacts ────────────────────────────────────────────────────────
    if not os.path.isfile(VECTORIZER_PKL):
        raise FileNotFoundError(
            "TF-IDF vectorizer not found. Run preprocessing.py first."
        )
    vectorizer = joblib.load(VECTORIZER_PKL)

    if not os.path.isfile(TRAIN_CSV):
        raise FileNotFoundError("train.csv not found. Run data_splitter.py first.")
    train_df = pd.read_csv(TRAIN_CSV)

    if not os.path.isfile(VAL_CSV):
        raise FileNotFoundError("val.csv not found. Run data_splitter.py first.")
    val_df = pd.read_csv(VAL_CSV)

    # ── Train / Load models ───────────────────────────────────────────────────
    train_matrix = (
        joblib.load(TRAIN_MATRIX_PKL)
        if os.path.isfile(TRAIN_MATRIX_PKL)
        else vectorizer.transform(
            train_df.apply(
                lambda r: str(r.get("article", ""))
                + " "
                + " ".join(str(r.get(c, "")) for c in ["opa", "opb", "opc", "opd"]),
                axis=1,
            ).tolist()
        )
    )

    verifier = train_verifier(train_df, train_matrix, vectorizer)
    kmeans   = train_kmeans(train_df, vectorizer)

    # ── Evaluate question extraction ──────────────────────────────────────────
    scores = evaluate_extraction(val_df, vectorizer)

    # ── Persist evaluation scores for UI ─────────────────────────────────────
    scores_path = os.path.join(MODELS_DIR, "model_a_scores.pkl")
    joblib.dump(scores, scores_path)
    print(f"[✓] Scores saved → {scores_path}")


if __name__ == "__main__":
    main()
