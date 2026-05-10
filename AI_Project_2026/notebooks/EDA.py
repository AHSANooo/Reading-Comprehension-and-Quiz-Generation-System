import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if not os.path.basename(BASE_DIR) == "AI_Project_2026":
    BASE_DIR = os.getcwd() # Fallback

DATA_PATH = os.path.join(BASE_DIR, "processed", "train.csv")

if not os.path.exists(DATA_PATH):
    print(f"Data not found at {DATA_PATH}. Please run src/data_splitter.py first.")
    exit()

df = pd.read_csv(DATA_PATH)

# Set visual style
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)

print("--- Dataset Overview ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 1. Answer Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='answer', data=df, order=['A', 'B', 'C', 'D'])
plt.title("Distribution of Correct Answers (A, B, C, D)")
plt.savefig(os.path.join(BASE_DIR, "notebooks", "answer_dist.png"))
plt.show()

# 2. Article Length Distribution
df['article_len'] = df['article'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.histplot(df['article_len'], bins=50, kde=True, color='blue')
plt.title("Article Word Count Distribution")
plt.xlabel("Number of Words")
plt.savefig(os.path.join(BASE_DIR, "notebooks", "article_len_dist.png"))
plt.show()

# 3. Question Length Distribution
df['question_len'] = df['question'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.histplot(df['question_len'], bins=30, kde=True, color='green')
plt.title("Question Word Count Distribution")
plt.xlabel("Number of Words")
plt.savefig(os.path.join(BASE_DIR, "notebooks", "question_len_dist.png"))
plt.show()

# 4. Word Cloud for Articles
text = " ".join(df['article'].sample(1000).astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Top Keywords in Articles")
plt.savefig(os.path.join(BASE_DIR, "notebooks", "wordcloud.png"))
plt.show()

print("\n--- Summary Statistics ---")
print(df[['article_len', 'question_len']].describe())
