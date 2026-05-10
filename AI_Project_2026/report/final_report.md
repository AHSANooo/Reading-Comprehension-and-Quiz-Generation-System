# Intelligent Reading Comprehension and Quiz Generation System
**Course:** AL2002 Artificial Intelligence  
**Team Members:** Muhammad Ahsan (23I-0553), Ali Ahmad Riaz (23I-2528)  
**Date:** May 10, 2026

---

## 1. Abstract
This project presents an end-to-end AI system for automated reading comprehension and quiz generation using the RACE dataset. The system integrates a traditional ML ensemble for answer verification, a K-Means clustering model for question categorization, and a multi-stage pipeline for distractor and hint generation. Our solution achieves high-quality quiz items by combining TF-IDF cosine similarity with POS-tagged noun chunk extraction and semantic Word2Vec ranking.

## 2. Introduction & Motivation
Automating the creation of educational assessments is a challenging task in Natural Language Processing. Traditional manual creation of distractors and hints is time-consuming. This project aims to leverage Classical Machine Learning techniques to generate pedagogically useful quizzes that include plausible distractors and graduated hints, enhancing the learning experience through automated feedback.

## 3. Related Work
- **RACE Dataset (Lai et al., 2017):** A large-scale reading comprehension dataset from examinations.
- **TF-IDF & Cosine Similarity:** Standard baselines for extractive question generation.
- **Word2Vec (Mikolov et al., 2013):** Used for semantic similarity in distractor generation.
- **Ensemble Learning:** Proven to improve classification robustness in answer verification.

## 4. Dataset Analysis
The RACE dataset contains ~100,000 questions. Our analysis (see `notebooks/EDA.py`) revealed:
- **Article Length:** Most passages are between 200-400 words.
- **Answer Balance:** Uniform distribution across A, B, C, and D options.
- **Keywords:** High frequency of proper nouns and descriptive adjectives, which we target for "fill-in-the-blank" generation.

## 5. Model A: Design, Training, Results
### 5.1 Design
- **Extraction:** TF-IDF Cosine Similarity identifies the most relevant sentence. NLTK POS tagging extracts the key noun chunk as the answer.
- **Verification:** A **Soft Voting Ensemble** combining Logistic Regression (C=0.1) and SGDClassifier.
- **Clustering:** Mini-Batch K-Means (k=10) to organize Q-A pairs by theme.

### 5.2 Results
- **Ensemble Accuracy:** ~75% on validation set.
- **Clustering Silhouette Score:** ~0.12 (standard for high-dimensional text data).
- **Extraction BLEU:** ~0.02 (extractive vs. abstractive gold questions).

## 6. Model B: Design, Training, Results
### 6.1 Design
- **Distractor Ranking:** Noun phrases are extracted from the article. A **Logistic Regression Ranker** scores candidates based on similarity to the correct answer, character overlap, and passage frequency.
- **Hint Engine:** Three graduated levels:
    1. **General:** Broad context sentence.
    2. **Specific:** Keyword overlap sentence.
    3. **Near-Explicit:** Answer sentence (masked).

### 6.2 Results
- **Distractor BLEU:** ~0.44 (highly plausible candidates).
- **Distractor ROUGE-L:** ~0.62.

## 7. User Interface Description
The system is deployed via **Streamlit** with a multi-screen design:
- **Quiz Studio:** Paste an article or load a **Random Sample** for instant quiz generation.
- **Quiz View:** Interactive radio buttons, color-coded feedback, and a **Graduated Hint Panel**.
- **Analytics Dashboard:** Real-time performance tracking (BLEU/ROUGE), latency metrics, and **CSV Export** for session logs.

## 8. Evaluation & Discussion
The ensemble verifier effectively filters out noisy generated questions. The transition from pure Word2Vec neighbors to ML-ranked passage candidates (Model B) significantly improved the "correctness" of distractors by ensuring they remain grounded in the article's vocabulary.

## 9. Limitations & Future Work
- **Syntactic Fluency:** Fill-in-the-blank templates occasionally produce awkward grammar.
- **Semantic Nuance:** Traditional ML lacks the deep semantic understanding of LLMs.
- **Future Work:** Integrating BERT embeddings for better candidate ranking and more diverse hint templates.

## 10. Conclusion
We successfully built a functional, modular AI pipeline that meets all university requirements. The system demonstrates the power of combining traditional ML classifiers with rule-based NLP for practical educational tools.

## 11. References
1. Lai, G. et al. (2017). RACE: Large-scale ReAding Comprehension Dataset. EMNLP.
2. Mikolov, T. et al. (2013). Distributed Representations of Words and Phrases. NIPS.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
