"""
preprocessing.py
----------------
Loads train.csv, fits a TF-IDF vectorizer over the article text, and
persists/loads the fitted vectorizer via a joblib checkpoint.

Returns both the vectorizer and the sparse TF-IDF matrix so downstream
models can use scipy.sparse structures directly (RAM-efficient).
"""

import os
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = "/content/drive/MyDrive/AI_Project_2026"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

TRAIN_CSV          = os.path.join(PROCESSED_DIR, "train.csv")
VECTORIZER_PKL     = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
TRAIN_MATRIX_PKL   = os.path.join(MODELS_DIR, "tfidf_train_matrix.pkl")


def _combine_text(row: pd.Series) -> str:
    """Concatenate article + options into a single string for the vectorizer."""
    options = " ".join(
        str(row.get(col, ""))
        for col in ["opa", "opb", "opc", "opd"]
    )
    return f"{row.get('article', '')} {options}"


def get_vectorizer(force_refit: bool = False) -> TfidfVectorizer:
    """
    Return a fitted TfidfVectorizer.

    Checkpointing
    -------------
    If ``tfidf_vectorizer.pkl`` already exists on Drive (and
    ``force_refit`` is False), the saved vectorizer is loaded and
    returned immediately — no data is re-read.

    Otherwise the vectorizer is fitted on ``train.csv`` and saved.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Checkpoint: load ─────────────────────────────────────────────────────
    if os.path.isfile(VECTORIZER_PKL) and not force_refit:
        print(f"[✓] Vectorizer checkpoint found — loading from: {VECTORIZER_PKL}")
        vectorizer = joblib.load(VECTORIZER_PKL)
        return vectorizer

    # ── Fit ──────────────────────────────────────────────────────────────────
    print(f"[→] Fitting TF-IDF vectorizer on: {TRAIN_CSV}")
    if not os.path.isfile(TRAIN_CSV):
        raise FileNotFoundError(
            f"train.csv not found at '{TRAIN_CSV}'.\n"
            "Run data_splitter.py first."
        )

    train_df = pd.read_csv(TRAIN_CSV)
    print(f"    Rows loaded: {len(train_df):,}")

    corpus = train_df.apply(_combine_text, axis=1).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        max_features=50_000,    # cap vocabulary for memory efficiency
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=2,               # ignore very rare terms
    )
    vectorizer.fit(corpus)
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # ── Checkpoint: save ─────────────────────────────────────────────────────
    joblib.dump(vectorizer, VECTORIZER_PKL)
    print(f"[✓] Vectorizer saved → {VECTORIZER_PKL}")

    return vectorizer


def get_train_matrix(
    vectorizer: TfidfVectorizer | None = None,
    force_recompute: bool = False,
) -> tuple[TfidfVectorizer, sp.csr_matrix, pd.DataFrame]:
    """
    Return (vectorizer, sparse_tfidf_matrix, train_df).

    The sparse matrix is cached as a separate pkl so it never has to be
    re-computed when the notebook kernel restarts.
    """
    if vectorizer is None:
        vectorizer = get_vectorizer()

    # ── Checkpoint: matrix ───────────────────────────────────────────────────
    if os.path.isfile(TRAIN_MATRIX_PKL) and not force_recompute:
        print(f"[✓] TF-IDF matrix checkpoint found — loading from: {TRAIN_MATRIX_PKL}")
        train_matrix = joblib.load(TRAIN_MATRIX_PKL)
        train_df = pd.read_csv(TRAIN_CSV)
        return vectorizer, train_matrix, train_df

    train_df = pd.read_csv(TRAIN_CSV)
    corpus   = train_df.apply(_combine_text, axis=1).tolist()

    print("[→] Transforming train corpus to TF-IDF matrix …")
    train_matrix = vectorizer.transform(corpus)          # scipy.sparse.csr_matrix
    print(f"    Matrix shape : {train_matrix.shape}")
    print(f"    Stored as    : {train_matrix.format.upper()} sparse matrix")

    joblib.dump(train_matrix, TRAIN_MATRIX_PKL)
    print(f"[✓] TF-IDF matrix saved → {TRAIN_MATRIX_PKL}")

    return vectorizer, train_matrix, train_df


# ── Stand-alone run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    vec, mat, df = get_train_matrix()
    print(f"\nVectorizer  : {vec}")
    print(f"Matrix type : {type(mat)}")
    print(f"Train rows  : {len(df):,}")
