# AI Reading Comprehension & Quiz Generation System

> Classical ML pipeline (TF-IDF · Logistic Regression · K-Means · Cosine Similarity)  
> Dataset: RACE · Evaluation: BLEU, ROUGE, METEOR

---

## Project Structure

```
AI_Project_2026/
├── dataset/
│   └── dev.csv                   ← Upload to Google Drive
├── processed/                    ← Auto-created by data_splitter.py
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/                       ← Auto-created; all .pkl checkpoints land here
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_train_matrix.pkl
│   ├── verifier_model.pkl
│   ├── kmeans_model.pkl
│   ├── model_a_scores.pkl
│   └── model_b_scores.pkl
├── src/
│   ├── data_splitter.py
│   ├── preprocessing.py
│   ├── model_a_train.py
│   └── model_b_train.py
├── ui/
│   └── app.py
└── requirements.txt
```

---

## Execution Order (Google Colab)

```python
# 1. Mount Drive and install dependencies (run once per session)
!pip install -q -r /content/drive/MyDrive/AI_Project_2026/requirements.txt

# 2. Split the dataset
%run /content/drive/MyDrive/AI_Project_2026/src/data_splitter.py

# 3. Fit TF-IDF vectorizer + build sparse matrix
%run /content/drive/MyDrive/AI_Project_2026/src/preprocessing.py

# 4. Train verifier + K-Means; evaluate question extraction
%run /content/drive/MyDrive/AI_Project_2026/src/model_a_train.py

# 5. Evaluate distractor generation
%run /content/drive/MyDrive/AI_Project_2026/src/model_b_train.py

# 6. Launch Streamlit UI (Colab tunnel)
!pip install -q streamlit pyngrok
from pyngrok import ngrok
import subprocess, time
proc = subprocess.Popen(
    ["streamlit", "run",
     "/content/drive/MyDrive/AI_Project_2026/ui/app.py",
     "--server.port", "8501"],
)
time.sleep(4)
tunnel = ngrok.connect(8501)
print("🚀 App URL:", tunnel.public_url)
```

> **Re-runs after Colab disconnect** — every script checks for `.pkl` files
> before doing any heavy computation.  If the checkpoints exist, they are
> loaded instantly.

---

## Architecture

| Module | Technique | Output |
|--------|-----------|--------|
| `preprocessing.py` | TF-IDF (`sublinear_tf`, bigrams, 50k vocab) | `tfidf_vectorizer.pkl` |
| `model_a_train.py` | Cosine Similarity (question extraction) | extracted question sentence |
| `model_a_train.py` | Logistic Regression (`saga` solver) | `verifier_model.pkl` |
| `model_a_train.py` | Mini-Batch K-Means (k=10) | `kmeans_model.pkl` |
| `model_b_train.py` | Cosine Similarity (top-N sentences) | hint list |
| `model_b_train.py` | NP-chunking + similarity filtering | 3 distractors |

### Evaluation Metrics

**No accuracy / precision / recall / F1 used.**  All text quality is
measured with:
- **BLEU** — n-gram precision with brevity penalty (`nltk`)
- **ROUGE-1 / ROUGE-2 / ROUGE-L** — recall-oriented overlap (`rouge-score`)
- **METEOR** — synonym-aware unigram alignment (`nltk`)

---

## Streamlit UI Features

- 📝 **Quiz Studio** — paste any article to extract a question + 4 options
- 💡 **Hint Panel** — collapsible; shows top-3 supporting sentences
- 📊 **Analytics Dashboard** — gauge cards for all 5 metrics per model
- ⚡ `@st.cache_resource` prevents re-loading `.pkl` files on every click
- 🔄 `st.session_state` tracks quiz flow (generated → answered → reset)
