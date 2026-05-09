"""
model_a_train.py
----------------
Model A  —  Fill-in-the-Blank Extractor  +  Soft Voting Verifier  +  K-Means Clusterer

Components
----------
1. Question Extractor (fill-in-the-blank)
   Selects the sentence from the article most similar to the correct answer
   via TF-IDF cosine similarity.  A key noun chunk within that sentence is
   identified via POS tagging and replaced with "__________".

2. Verifier (Soft Voting Ensemble)
   Three classifiers vote with probability averaging:
     - Logistic Regression  (C tuned via GridSearchCV)
     - Multinomial Naive Bayes
     - SGDClassifier  (loss='log_loss')
   Saved as  models/verifier_ensemble_model.pkl

3. Clusterer (Mini-Batch K-Means)
   Clusters question-answer pair vectors for thematic organisation.
   Saved as  models/kmeans_model.pkl

4. Evaluation  (val.csv)
   BLEU, ROUGE-1/2/L, METEOR comparing extracted stems to gold questions.

Checkpointing
-------------
Each heavy artefact is persisted as .pkl and skipped on subsequent runs.
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore")


for _pkg in ("wordnet", "punkt", "punkt_tab", "omw-1.4",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    nltk.download(_pkg, quiet=True)


BASE_DIR = "/content/drive/MyDrive/AI_Project_2026"
if not os.path.isdir(BASE_DIR):
    LOCAL_BASE = "/home/ahsan/Documents/Uni work/Sem 6/AI Lab/Project/AI_Project_2026"
    if os.path.isdir(LOCAL_BASE):
        BASE_DIR = LOCAL_BASE
    else:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

TRAIN_CSV        = os.path.join(PROCESSED_DIR, "train.csv")
VAL_CSV          = os.path.join(PROCESSED_DIR, "val.csv")
VECTORIZER_PKL   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
TRAIN_MATRIX_PKL = os.path.join(MODELS_DIR, "tfidf_train_matrix.pkl")
ENSEMBLE_PKL     = os.path.join(MODELS_DIR, "verifier_ensemble_model.pkl")
KMEANS_PKL       = os.path.join(MODELS_DIR, "kmeans_model.pkl")

N_CLUSTERS      = 10
RANDOM_STATE    = 42
BLANK           = "__________"


GENERIC_WORD_BLACKLIST = {
    "something", "anything", "nothing", "everything", "things",
    "someone",   "anyone",   "it",      "this",       "that",
    "they",      "he",       "she",     "one",        "ones",
    "way",       "part",     "kind",    "type",       "lot",
}


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


def _chunk_is_valid(chunk: str, pos_tagged_leaves: list) -> bool:
    if len(chunk) < 4:
        return False

    chunk_words = chunk.lower().split()

    if all(w in GENERIC_WORD_BLACKLIST for w in chunk_words):
        return False

    if chunk == chunk.lower():
        return False

    return True


def _chunk_priority(chunk: str, pos_tagged_leaves: list) -> tuple:
    chunk_words_lower = set(chunk.lower().split())

    has_proper_noun = any(
        word.lower() in chunk_words_lower and tag == "NNP"
        for word, tag in pos_tagged_leaves
    )

    word_count = len(chunk.split())

    return (int(has_proper_noun), word_count, len(chunk))


def _extract_key_noun_chunk(sentence: str) -> str:
    tokens     = nltk.word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(tokens)
    grammar    = r"NP: {<NNP>|<NN>}"
    parser     = nltk.RegexpParser(grammar)
    tree       = parser.parse(pos_tagged)

    raw_chunks = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        leaves = subtree.leaves()
        chunk_text = " ".join(word for word, _ in leaves)
        if len(chunk_text.split()) == 1:
            raw_chunks.append(chunk_text)

    valid_chunks = [c for c in raw_chunks if _chunk_is_valid(c, pos_tagged)]
    if not valid_chunks:
        return ""
    return max(valid_chunks, key=lambda c: _chunk_priority(c, pos_tagged))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Question Extraction — cosine similarity + fill-in-the-blank
# ══════════════════════════════════════════════════════════════════════════════

def extract_question(
    article: str,
    correct_answer: str,
    vectorizer,
) -> tuple[str, str]:
    """
    Return (question_stem, answer_chunk).

    The best article sentence is selected by TF-IDF cosine similarity to
    correct_answer.  The most prominent noun chunk in that sentence is
    replaced with BLANK to form a fill-in-the-blank question.
    """
    sents = _sentences(article)
    if not sents:
        return ("", "")

    corpus    = sents + [correct_answer]
    tfidf_mat = vectorizer.transform(corpus)

    sent_vecs  = tfidf_mat[:-1]
    answer_vec = tfidf_mat[-1]

    sims          = cosine_similarity(sent_vecs, answer_vec).flatten()
    best_sentence = sents[int(np.argmax(sims))]

    key_chunk = _extract_key_noun_chunk(best_sentence)
    if not key_chunk:
        return (best_sentence + f" {BLANK}?", correct_answer)

    question_stem = re.sub(
        re.escape(key_chunk), BLANK, best_sentence, count=1, flags=re.IGNORECASE
    )
    return (question_stem, key_chunk)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Verifier — Soft Voting Ensemble
# ══════════════════════════════════════════════════════════════════════════════

def _build_verifier_dataset(
    df: pd.DataFrame, vectorizer
) -> tuple[sp.csr_matrix, np.ndarray]:
    texts, labels = [], []
    option_cols   = ["A", "B", "C", "D"]
    answer_map    = {"A": "A", "B": "B", "C": "C", "D": "D"}

    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        correct = answer_map.get(str(row.get("answer", "A")).upper(), "A")
        for col in option_cols:
            texts.append(question + " " + str(row.get(col, "")))
            labels.append(1 if col == correct else 0)

    return vectorizer.transform(texts), np.array(labels, dtype=np.int8)


def train_ensemble_verifier(
    train_df: pd.DataFrame,
    vectorizer,
    force_retrain: bool = False,
) -> VotingClassifier:
    """
    Build and save the soft-voting ensemble.

    Logistic Regression C is selected via 3-fold GridSearchCV (F1 scoring).
    Multinomial Naive Bayes and SGDClassifier are added with default-sensible
    hyperparameters.  All three vote using averaged class probabilities.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.isfile(ENSEMBLE_PKL) and not force_retrain:
        print(f"[✓] Ensemble checkpoint found — loading: {ENSEMBLE_PKL}")
        return joblib.load(ENSEMBLE_PKL)

    print("[→] Building verifier training dataset …")
    X_train, y_train = _build_verifier_dataset(train_df, vectorizer)
    print(f"    Samples : {X_train.shape[0]:,}  |  Features : {X_train.shape[1]:,}")
    print(f"    Positive: {y_train.sum():,}  |  Negative: {(y_train == 0).sum():,}")

    print("[→] Running GridSearchCV for Logistic Regression C …")
    gs = GridSearchCV(
        LogisticRegression(
            max_iter=1_000,
            solver="saga",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        {"C": [0.1, 1.0, 10.0]},
        cv=3, scoring="accuracy", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    best_lr = gs.best_estimator_
    print(f"    Best C selected: {gs.best_params_['C']}")

    sgd = SGDClassifier(
        loss="log_loss",
        max_iter=1_000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[("lr", best_lr), ("sgd", sgd)],
        voting="soft",
        n_jobs=-1,
    )

    print("[→] Fitting Soft Voting Ensemble …")
    ensemble.fit(X_train, y_train)

    joblib.dump(ensemble, ENSEMBLE_PKL)
    print(f"[✓] Ensemble verifier saved → {ENSEMBLE_PKL}")
    return ensemble


# ══════════════════════════════════════════════════════════════════════════════
# 2c. Verifier Inference — fill-blank-then-score
# ══════════════════════════════════════════════════════════════════════════════

def verify_option(
    question_stem: str,
    option_text: str,
    article: str,
    vectorizer,
    ensemble: VotingClassifier,
) -> tuple[int, float]:
    """
    Return (predicted_label, confidence_probability).

    The model is evaluated using the question and the option.
    This provides a much higher signal-to-noise ratio than prepending
    the entire article, allowing the ensemble to properly distinguish
    and output meaningful confidence scores.
    """
    feature_text = question_stem + " " + option_text

    vec   = vectorizer.transform([feature_text])
    proba = ensemble.predict_proba(vec)[0]

    predicted_label       = int(np.argmax(proba))
    confidence_of_correct = float(proba[1])

    return predicted_label, confidence_of_correct


# ══════════════════════════════════════════════════════════════════════════════
# 2b. Verifier Evaluation — Classification Report & Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_verifier(
    val_df: pd.DataFrame,
    vectorizer,
    ensemble: VotingClassifier,
) -> None:
    """
    Evaluate the trained ensemble on the validation split.

    Builds the same (article + option, label) dataset from val_df, runs
    inference, then prints a full classification_report and a
    confusion_matrix so class-imbalance effects are clearly visible.
    """
    print("\n[→] Building verifier validation dataset …")
    X_val, y_val = _build_verifier_dataset(val_df, vectorizer)
    print(f"    Val samples : {X_val.shape[0]:,}")


    y_pred = ensemble.predict(X_val)


    print("\n" + "═" * 60)
    print("  Verifier Ensemble — Classification Report (val split)")
    print("═" * 60)
    print(classification_report(y_val, y_pred, target_names=["Incorrect", "Correct"]))


    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion Matrix")
    print("  (rows = Actual, cols = Predicted  |  0=Incorrect, 1=Correct)")
    print("\n  " + str(cm).replace("\n", "\n  "))
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Clusterer — Mini-Batch K-Means
# ══════════════════════════════════════════════════════════════════════════════

def _build_cluster_matrix(df: pd.DataFrame, vectorizer) -> sp.csr_matrix:
    texts = [
        str(row.get("question", "")) + " " + _correct_option(row)
        for _, row in df.iterrows()
    ]
    return vectorizer.transform(texts)


def train_kmeans(
    train_df: pd.DataFrame,
    vectorizer,
    force_retrain: bool = False,
) -> MiniBatchKMeans:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.isfile(KMEANS_PKL) and not force_retrain:
        print(f"[✓] K-Means checkpoint found — loading: {KMEANS_PKL}")
        return joblib.load(KMEANS_PKL)

    print("[→] Building clustering matrix …")
    X_cluster = _build_cluster_matrix(train_df, vectorizer)
    print(f"    Matrix shape: {X_cluster.shape}")

    print(f"[→] Fitting Mini-Batch K-Means (k={N_CLUSTERS}) …")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
                         batch_size=1_024, n_init=10)
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
    vec = vectorizer.transform([question + " " + answer])
    return int(kmeans.predict(vec)[0])


# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluation — BLEU, ROUGE, METEOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_extraction(val_df: pd.DataFrame, vectorizer) -> dict:
    smoother = SmoothingFunction().method1
    rscorer  = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    bleu_scores, rouge1_scores, rouge2_scores = [], [], []
    rougeL_scores, meteor_scores = [], []

    print(f"\n[→] Evaluating question extraction on {len(val_df):,} val examples …")

    for _, row in val_df.iterrows():
        article = str(row.get("article", ""))
        gold_q  = str(row.get("question", ""))
        answer  = _correct_option(row)

        if not article.strip() or not gold_q.strip():
            continue

        question_stem, _ = extract_question(article, answer, vectorizer)
        ref = [_tokenize(gold_q)]
        hyp = _tokenize(question_stem)

        bleu_scores.append(sentence_bleu(ref, hyp, smoothing_function=smoother))

        r = rscorer.score(gold_q, question_stem)
        rouge1_scores.append(r["rouge1"].fmeasure)
        rouge2_scores.append(r["rouge2"].fmeasure)
        rougeL_scores.append(r["rougeL"].fmeasure)
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

    ensemble = train_ensemble_verifier(train_df, vectorizer)
    evaluate_verifier(val_df, vectorizer, ensemble)


    train_kmeans(train_df, vectorizer)

    scores = evaluate_extraction(val_df, vectorizer)

    scores_path = os.path.join(MODELS_DIR, "model_a_scores.pkl")
    joblib.dump(scores, scores_path)
    print(f"[✓] Scores saved → {scores_path}")


if __name__ == "__main__":
    main()
