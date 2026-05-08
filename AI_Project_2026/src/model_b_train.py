"""
model_b_train.py
----------------
Model B  —  Distractor Generator  +  Hint Engine

Components
----------
1. Hint Generator
   Ranks every sentence in the article by cosine similarity to the
   question (TF-IDF space).  Top-N sentences are returned as hints.

2. Distractor Generator
   Extracts noun-phrase / key-word candidates from the article using
   NLTK chunking, computes cosine similarity against the correct answer,
   and returns the 3 candidates with medium-to-low similarity
   (i.e. related to the article context but clearly NOT the answer).

3. Evaluation
   Compares the generated distractors to the actual wrong options in
   val.csv using BLEU, ROUGE-1/2/L, and METEOR.  Results are printed.

Checkpointing
-------------
Model B is stateless (no trainable parameters) — all computation is
done at inference time.  Evaluation scores are saved as
models/model_b_scores.pkl for the UI to display.
"""

import os
import warnings

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Download required NLTK data ───────────────────────────────────────────────
for _pkg in ("wordnet", "punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
             "maxent_ne_chunker_tab", "words", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = "/content/drive/MyDrive/AI_Project_2026"
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

VAL_CSV        = os.path.join(PROCESSED_DIR, "val.csv")
VECTORIZER_PKL = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SCORES_PKL     = os.path.join(MODELS_DIR, "model_b_scores.pkl")

# Hint / distractor tuning
TOP_N_HINTS         = 3
N_DISTRACTORS       = 3
DISTRACTOR_SIM_LOW  = 0.05   # similarity lower-bound (must be *somewhat* related)
DISTRACTOR_SIM_HIGH = 0.55   # similarity upper-bound (must NOT be the answer)
MIN_CANDIDATE_LEN   = 3      # minimum characters for a candidate phrase


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
    mapping = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
    correct_col = mapping.get(str(row.get("answer", "A")).upper(), "opa")
    return [
        str(row.get(col, ""))
        for col in ["opa", "opb", "opc", "opd"]
        if col != correct_col
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Hint Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_hints(
    question: str,
    article: str,
    vectorizer,
    top_n: int = TOP_N_HINTS,
) -> list[str]:
    """
    Return the top-N article sentences most similar to the question.
    Similarity is measured in TF-IDF cosine-similarity space.
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
# 2. Distractor Generator
# ══════════════════════════════════════════════════════════════════════════════

def _extract_candidates(article: str) -> list[str]:
    """
    Extract noun-phrase candidates from *article* using NLTK POS-tagging
    and a simple NP chunk grammar.  Falls back to individual nouns if
    the chunker yields nothing useful.
    """
    tokens     = nltk.word_tokenize(article)
    pos_tagged = nltk.pos_tag(tokens)

    # Simple noun-phrase grammar: optional adjectives followed by nouns
    grammar = r"NP: {<JJ>*<NN.*>+}"
    cp      = nltk.RegexpParser(grammar)
    tree    = cp.parse(pos_tagged)

    candidates = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join(word for word, _ in subtree.leaves())
        if len(phrase) >= MIN_CANDIDATE_LEN:
            candidates.append(phrase)

    # Fallback: individual nouns
    if not candidates:
        candidates = [
            word for word, tag in pos_tagged
            if tag.startswith("NN") and len(word) >= MIN_CANDIDATE_LEN
        ]

    return list(dict.fromkeys(candidates))      # deduplicate, preserve order


def generate_distractors(
    article: str,
    correct_answer: str,
    vectorizer,
    n: int = N_DISTRACTORS,
) -> list[str]:
    """
    Return *n* distractor phrases from *article*.

    Strategy
    --------
    1. Extract noun-phrase candidates from the article.
    2. Compute cosine similarity of each candidate against the correct answer.
    3. Keep candidates whose similarity falls in (DISTRACTOR_SIM_LOW, DISTRACTOR_SIM_HIGH):
       — related enough to the topic, but clearly not the answer.
    4. If fewer than *n* remain after filtering, relax the bounds and fall
       back to the lowest-similarity candidates.
    """
    candidates = _extract_candidates(article)
    if not candidates:
        return []

    corpus    = candidates + [correct_answer]
    tfidf_mat = vectorizer.transform(corpus)

    cand_vecs  = tfidf_mat[:-1]
    answer_vec = tfidf_mat[-1]

    sims = cosine_similarity(cand_vecs, answer_vec).flatten()

    # Filter to medium/low similarity band
    mask = (sims >= DISTRACTOR_SIM_LOW) & (sims <= DISTRACTOR_SIM_HIGH)
    filtered = [(candidates[i], sims[i]) for i in range(len(candidates)) if mask[i]]

    # Sort by descending similarity (most plausible first)
    filtered.sort(key=lambda x: x[1], reverse=True)

    distractors = [phrase for phrase, _ in filtered[:n]]

    # Fallback: if not enough candidates in the band, take the lowest-sim ones
    if len(distractors) < n:
        fallback = sorted(
            [(candidates[i], sims[i]) for i in range(len(candidates))],
            key=lambda x: x[1],
        )
        for phrase, _ in fallback:
            if phrase not in distractors:
                distractors.append(phrase)
            if len(distractors) == n:
                break

    return distractors[:n]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Evaluation — BLEU, ROUGE, METEOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_distractors(val_df: pd.DataFrame, vectorizer) -> dict:
    """
    For each row in val_df:
      - generate distractors from the article + correct answer
      - compare them to the actual wrong options
    Aggregate BLEU, ROUGE-1/2/L, METEOR; print and return scores.

    Matching strategy: for each generated distractor, find the best-matching
    gold wrong option and average the scores.
    """
    smoother = SmoothingFunction().method1
    rscorer  = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    bleu_scores   = []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    meteor_scores = []

    print(f"\n[→] Evaluating distractor generation on {len(val_df):,} val examples …")

    for _, row in val_df.iterrows():
        article = str(row.get("article", ""))
        correct = _correct_option(row)
        wrongs  = _wrong_options(row)

        if not article.strip():
            continue

        generated = generate_distractors(article, correct, vectorizer)
        if not generated:
            continue

        for gen_dist in generated:
            hyp = _tokenize(gen_dist)
            if not hyp:
                continue

            # Best match against any gold wrong option
            best_bleu, best_r1, best_r2, best_rL, best_met = 0.0, 0.0, 0.0, 0.0, 0.0
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
    # ── Checkpoint: skip evaluation if scores already exist ───────────────────
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

    # ── Load dependencies ─────────────────────────────────────────────────────
    if not os.path.isfile(VECTORIZER_PKL):
        raise FileNotFoundError(
            "TF-IDF vectorizer not found. Run preprocessing.py first."
        )
    vectorizer = joblib.load(VECTORIZER_PKL)

    if not os.path.isfile(VAL_CSV):
        raise FileNotFoundError("val.csv not found. Run data_splitter.py first.")
    val_df = pd.read_csv(VAL_CSV)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    scores = evaluate_distractors(val_df, vectorizer)

    # ── Save scores for UI ────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scores, SCORES_PKL)
    print(f"[✓] Scores saved → {SCORES_PKL}")


if __name__ == "__main__":
    main()
