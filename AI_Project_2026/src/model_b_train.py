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

warnings.filterwarnings("ignore")


for _pkg in ("wordnet", "punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
             "maxent_ne_chunker_tab", "words", "omw-1.4"):
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

TRAIN_CSV      = os.path.join(PROCESSED_DIR, "train.csv")
VAL_CSV        = os.path.join(PROCESSED_DIR, "val.csv")
VECTORIZER_PKL = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
W2V_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec.model")
SCORES_PKL     = os.path.join(MODELS_DIR, "model_b_scores.pkl")

TOP_N_HINTS         = 3
W2V_TOP_NEIGHBOURS  = 10
N_DISTRACTORS       = 3
MIN_CANDIDATE_LEN   = 3

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
    mapping = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
    col = mapping.get(str(row.get("answer", "A")).upper(), "opa")
    return str(row.get(col, ""))


def _wrong_options(row: pd.Series) -> list[str]:
    mapping     = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
    correct_col = mapping.get(str(row.get("answer", "A")).upper(), "opa")
    return [
        str(row.get(col, ""))
        for col in ["opa", "opb", "opc", "opd"]
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
    all_text_columns = ["article", "opa", "opb", "opc", "opd", "question"]
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
    top_n: int = TOP_N_HINTS,
) -> list[str]:
    """
    Return the top-N article sentences most similar to the question stem
    measured in TF-IDF cosine-similarity space.
    """
    sents = _sentences(article)
    if not sents:
        return []

    corpus    = sents + [question]
    tfidf_mat = vectorizer.transform(corpus)

    sent_vecs    = tfidf_mat[:-1]
    question_vec = tfidf_mat[-1]

    sims     = cosine_similarity(sent_vecs, question_vec).flatten()
    top_idxs = np.argsort(sims)[::-1][:top_n]
    return [sents[i] for i in top_idxs]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Distractor Generator — Word2Vec semantic neighbours
# ══════════════════════════════════════════════════════════════════════════════

def generate_distractors(
    article: str,
    correct_answer: str,
    w2v_model: Word2Vec,
    n: int = N_DISTRACTORS,
) -> list[str]:
    """
    Generate *n* distractors for *correct_answer* using Word2Vec.

    Strategy
    --------
    1. Retrieve the W2V_TOP_NEIGHBOURS nearest semantic neighbours of the
       correct answer (averaged word vector when multi-word phrase).
    2. Filter out neighbours whose surface form appears in the passage.
    3. Return the top *n* remaining neighbours.
    """
    answer_tokens     = _tokenize(correct_answer)
    passage_tokens    = set(_tokenize(article))

    vocab = w2v_model.wv

    answer_vectors = [
        vocab[token]
        for token in answer_tokens
        if token in vocab
    ]

    distractors = []

    if answer_vectors:
        mean_vector = np.mean(answer_vectors, axis=0)
        raw_neighbours = vocab.similar_by_vector(mean_vector, topn=W2V_TOP_NEIGHBOURS)
        
        for word, _ in raw_neighbours:
            if len(distractors) >= n:
                break
            if word not in passage_tokens and len(word) >= 3:
                if re.match(r"^[a-zA-Z\s]+$", word):
                    distractors.append(word)

    if len(distractors) < n:
        try:
            nltk_words = nltk.corpus.words.words()
        except LookupError:
            nltk.download("words", quiet=True)
            nltk_words = nltk.corpus.words.words()

        valid_fallback = [
            w for w in nltk_words 
            if len(w) >= 3 and w.lower() not in passage_tokens and re.match(r"^[a-zA-Z\s]+$", w)
        ]

        while len(distractors) < n and valid_fallback:
            fallback_word = random.choice(valid_fallback)
            if fallback_word not in distractors:
                distractors.append(fallback_word)

    return distractors


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

        if not article.strip():
            continue

        generated = generate_distractors(article, correct, w2v_model)
        if not generated:
            continue

        for gen_dist in generated:
            hyp = _tokenize(gen_dist)
            if not hyp:
                continue

            best_bleu = best_r1 = best_r2 = best_rL = best_met = 0.0
            for gold in wrongs:
                if not gold.strip():
                    continue
                ref = [_tokenize(gold)]
                b   = sentence_bleu(ref, hyp, smoothing_function=smoother)
                r   = rscorer.score(gold, gen_dist)
                m   = meteor_score(ref, hyp)
                if b > best_bleu:
                    best_bleu = b
                    best_r1   = r["rouge1"].fmeasure
                    best_r2   = r["rouge2"].fmeasure
                    best_rL   = r["rougeL"].fmeasure
                    best_met  = m

            bleu_scores.append(best_bleu)
            rouge1_scores.append(best_r1)
            rouge2_scores.append(best_r2)
            rougeL_scores.append(best_rL)
            meteor_scores.append(best_met)

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
    if os.path.isfile(SCORES_PKL):
        print(f"[✓] Evaluation checkpoint found — loading: {SCORES_PKL}")
        scores = joblib.load(SCORES_PKL)
        print("\n" + "═" * 50)
        print("  Model B — Distractor Generation Evaluation")
        print("═" * 50)
        for metric, score in scores.items():
            print(f"  {metric:<10} : {score:.4f}")
        print("═" * 50)
        return

    if not os.path.isfile(TRAIN_CSV):
        raise FileNotFoundError("train.csv not found. Run data_splitter.py first.")
    train_df = pd.read_csv(TRAIN_CSV)

    if not os.path.isfile(VAL_CSV):
        raise FileNotFoundError("val.csv not found. Run data_splitter.py first.")
    val_df = pd.read_csv(VAL_CSV)

    w2v_model = train_word2vec(train_df)

    scores = evaluate_distractors(val_df, w2v_model)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scores, SCORES_PKL)
    print(f"[✓] Scores saved → {SCORES_PKL}")


if __name__ == "__main__":
    main()
