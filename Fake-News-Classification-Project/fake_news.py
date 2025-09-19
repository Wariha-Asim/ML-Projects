# Fake News Classification Project (Clean .py version)

# ==============================
# Importing Libraries
# ==============================
import pandas as pd
import numpy as np

# Text preprocessing
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer

# Model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score
)

# Visualization
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# ==============================
# Load Data
# ==============================
df_fake_news = pd.read_csv('Fake.csv')
df_real_news = pd.read_csv('True.csv')

# Add labels
df_fake_news['label'] = 0
df_real_news['label'] = 1

# Combine datasets
df = pd.concat([df_fake_news, df_real_news], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# ==============================
# Preprocessing
# ==============================
df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace('[^a-zA-Z]', ' ', regex=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.lower() not in stop_words]
    return " ".join(tokens)

df['text'] = df['text'].apply(preprocess)

# ==============================
# Feature Extraction
# ==============================
X = df['text']
y = df['label']
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_counts, y, test_size=0.3, random_state=42
)

# ==============================
# Naive Bayes
# ==============================
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)
nb_pre = precision_score(y_test, nb_pred)
nb_rec = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)
nb_cm = confusion_matrix(y_test, nb_pred)
nb_roc = roc_auc_score(y_test, nb_pred)

# ==============================
# Logistic Regression
# ==============================
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_pre = precision_score(y_test, lr_pred)
lr_rec = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_cm = confusion_matrix(y_test, lr_pred)
lr_roc = roc_auc_score(y_test, lr_pred)

# ==============================
# Random Forest
# ==============================
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_pre = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
rf_roc = roc_auc_score(y_test, rf_pred)

# ==============================
# Model Comparison
# ==============================
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Naive Bayes"],
    "Accuracy": [lr_acc, rf_acc, nb_acc],
    "Precision": [lr_pre, rf_pre, nb_pre],
    "Recall": [lr_rec, rf_rec, nb_rec],
    "F1_score": [lr_f1, rf_f1, nb_f1],
    "ROC_AUC": [lr_roc, rf_roc, nb_roc]
})

print("============== 📊 Model Comparison Table ==============")
print(results.to_string())

# Ranking Models
results["Acc_Rank"] = results["Accuracy"].rank(ascending=False)
results["F1_Rank"] = results["F1_score"].rank(ascending=False)
results["ROC_AUC_Rank"] = results["ROC_AUC"].rank(ascending=False)
results["Overall_Rank"] = results[["Acc_Rank","F1_Rank","ROC_AUC_Rank"]].sum(axis=1)

results_sorted = results.sort_values("Overall_Rank")
print("\n============== 🏆 Ranked Models ==============")
print(results_sorted.to_string())
print("\n🏆 Best Model Selected:", results_sorted.iloc[0]["Model"])

# ==============================
# Visualizations
# ==============================

# Top 20 Words in Fake News
fake_words = " ".join(df[df['label'] == 0]['text']).split()
fake_counter = Counter(fake_words)
top_fake = fake_counter.most_common(20)

words_fake = [w for w, c in top_fake]
counts_fake = [c for w, c in top_fake]

plt.figure(figsize=(10,6))
plt.bar(words_fake, counts_fake, color='red')
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Words in Fake News")
plt.ylabel("Frequency")
plt.show()

# Top 20 Words in Real News
real_words = " ".join(df[df['label'] == 1]['text']).split()
real_counter = Counter(real_words)
top_real = real_counter.most_common(20)

words_real = [w for w, c in top_real]
counts_real = [c for w, c in top_real]

plt.figure(figsize=(10,6))
plt.bar(words_real, counts_real, color='green')
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Words in Real News")
plt.ylabel("Frequency")
plt.show()

# Histogram: Article Length
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10,5))
plt.hist(df[df['label'] == 0]['word_count'], bins=30, alpha=0.6, label='Fake News', color='pink')
plt.hist(df[df['label'] == 1]['word_count'], bins=30, alpha=0.6, label='Real News', color='purple')
plt.xlabel("Number of Words per Article")
plt.ylabel("Number of Articles")
plt.title("Article Length Distribution: Fake vs Real News", fontsize=16)
plt.legend()
plt.show()

# Wordclouds
positive_reviews = df[df['label']==1]['text']
negative_reviews = df[df['label']==0]['text']

positive_wordcloud = WordCloud(colormap='PuRd').generate(' '.join(positive_reviews))
negative_wordcloud = WordCloud(colormap='plasma').generate(' '.join(negative_reviews))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Real News')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Fake News')
plt.axis('off')

plt.tight_layout()
plt.show()
