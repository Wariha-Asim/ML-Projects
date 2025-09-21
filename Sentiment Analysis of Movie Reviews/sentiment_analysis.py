# ==============================
# Importing Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score
)
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv(r'C:\Users\AR FAST\Documents\Machine Learning mini projects\Sentiment Analysis of Movie Reviews\IMDB Dataset.csv')

# Encode sentiment (0 = negative, 1 = positive)
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# ==============================
# Data Preprocessing
# ==============================
print("\nBefore Preprocessing")
print(df.head(2))

def preprocess_text(text):
    text = re.sub('<.*?>', ' ', text)               # Remove HTML tags
    text = text.lower()                             # Lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)          # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()        # Remove extra spaces
    tokens = text.split()                           # Tokenization
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

df['review'] = df['review'].astype(str).apply(preprocess_text)

print("\nAfter Preprocessing")
print(df.head(5))

# ==============================
# Feature Extraction (CountVectorizer for NB & LR)
# ==============================
X = df['review']
y = df['sentiment']
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.3, random_state=42)

# ==============================
# Naive Bayes Model
# ==============================
print("\n========== Naive Bayes Model Prediction =================")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("First 10 Predictions:", nb_pred[:10])

nb_acc = accuracy_score(y_test, nb_pred)
nb_pre = precision_score(y_test, nb_pred)
nb_rec = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)
nb_cm = confusion_matrix(y_test, nb_pred)
nb_roc = roc_auc_score(y_test, nb_pred)

print(f"Accuracy: {nb_acc:.4f}")
print(f"Precision: {nb_pre:.4f}")
print(f"Recall: {nb_rec:.4f}")
print(f"F1-Score: {nb_f1:.4f}")
print("Confusion Matrix:\n", nb_cm)
print(f"ROC-AUC: {nb_roc:.4f}")

# ==============================
# Logistic Regression Model
# ==============================
print("\n========== Logistic Regression Model Prediction =================")
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("First 10 Predictions:", lr_pred[:10])

lr_acc = accuracy_score(y_test, lr_pred)
lr_pre = precision_score(y_test, lr_pred)
lr_rec = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_cm = confusion_matrix(y_test, lr_pred)
lr_roc = roc_auc_score(y_test, lr_pred)

print(f"Accuracy: {lr_acc:.4f}")
print(f"Precision: {lr_pre:.4f}")
print(f"Recall: {lr_rec:.4f}")
print(f"F1-Score: {lr_f1:.4f}")
print("Confusion Matrix:\n", lr_cm)
print(f"ROC-AUC: {lr_roc:.4f}")

# ==============================
# Support Vector Machine Model
# ==============================
print("\n========== Support Vector Machine Model Prediction =================")
df_sampled = df.sample(n=10000, random_state=42, replace=False)
X_sampled = df_sampled['review']
y_sampled = df_sampled['sentiment']

vectorizer = TfidfVectorizer(max_features=5000)
X_counts = vectorizer.fit_transform(X_sampled)

X_train, X_test, y_train, y_test = train_test_split(X_counts, y_sampled, test_size=0.3, random_state=42)

svm = SVC(random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("First 10 Predictions:", svm_pred[:10])
print("Actual first 10 labels:", y_test.values[:10])

svm_acc = accuracy_score(y_test, svm_pred)
svm_pre = precision_score(y_test, svm_pred)
svm_rec = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_cm = confusion_matrix(y_test, svm_pred)
svm_roc = roc_auc_score(y_test, svm_pred)

print(f"Accuracy: {svm_acc:.4f}")
print(f"Precision: {svm_pre:.4f}")
print(f"Recall: {svm_rec:.4f}")
print(f"F1-Score: {svm_f1:.4f}")
print("Confusion Matrix:\n", svm_cm)
print(f"ROC-AUC: {svm_roc:.4f}")

# ==============================
# Model Comparison
# ==============================
results = pd.DataFrame({
    'Model': ['Naive Bayes', 'Logistic Regression', 'Support Vector Machine'],
    'Accuracy': [nb_acc, lr_acc, svm_acc],
    'Precision': [nb_pre, lr_pre, svm_pre],
    'Recall': [nb_rec, lr_rec, svm_rec],
    'F1-Score': [nb_f1, lr_f1, svm_f1],
    'ROC-AUC': [nb_roc, lr_roc, svm_roc]
})

print("\n============== 📊 Model Comparison Table ==============")
print(results.to_string())

metrics_to_rank = ["Accuracy", "F1-Score", "ROC-AUC"]
for metric in metrics_to_rank:
    results[f"{metric}_Rank"] = results[metric].rank(ascending=False)
results["Overall_Rank"] = results[[f"{m}_Rank" for m in metrics_to_rank]].sum(axis=1)
results_sorted = results.sort_values("Overall_Rank")

print("\n============== 🏆 Ranked Models ==============")
print(results_sorted.to_string())
print("\n🏆 Best Model Selected:", results_sorted.iloc[0]["Model"])

# ==============================
# Confusion Matrix for Best Model (Logistic Regression)
# ==============================
svm_cm_percent = svm_cm.astype("float") / svm_cm.sum() * 100
plt.figure(figsize=(7, 8))
plt.imshow(svm_cm_percent, interpolation="nearest", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix (%)", fontsize=14, fontweight="bold")
plt.colorbar(label="Percentage")

plt.xticks([0, 1], ["Predicted Negative (0)", "Predicted Positive (1)"], fontsize=10)
plt.yticks([0, 1], ["Actual Negative (0)", "Actual Positive (1)"], fontsize=10)
plt.ylabel("Actual Class", fontsize=12)
plt.xlabel("Predicted Class", fontsize=12)

for i in range(svm_cm_percent.shape[0]):
    for j in range(svm_cm_percent.shape[1]):
        plt.text(j, i, f"{svm_cm_percent[i, j]:.2f}%", 
                 ha="center", va="center", color="black", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()

# ==============================
# WordCloud for Positive & Negative Reviews
# ==============================
positive_reviews = df[df['sentiment'] == 1]['review']
negative_reviews = df[df['sentiment'] == 0]['review']

positive_wordcloud = WordCloud(colormap='Greens').generate(' '.join(positive_reviews))
negative_wordcloud = WordCloud(colormap='Reds').generate(' '.join(negative_reviews))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Reviews')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Reviews')
plt.axis('off')

plt.tight_layout()
plt.show()

# ==============================
# Histogram – Review Length Distribution
# ==============================

# Calculate review lengths
df['review_length'] = df['review'].apply(lambda x: len(x.split())) #apply() lets you run a function on each element of a column
#for each row of x in text column split the string into words and count them.

# Separate lengths by sentiment
pos_lengths = df[df['sentiment']==1]['review_length']
neg_lengths = df[df['sentiment']==0]['review_length']

# Plot side-by-side histograms
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.hist(pos_lengths, bins=20, color='red', alpha=0.6, edgecolor='black')
plt.title('Positive Reviews Length')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(neg_lengths, bins=20, color='purple', alpha=0.6, edgecolor='black')
plt.title('Negative Reviews Length')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


## Distribution of Review Length vs Word Count Categories
#Bar Chart
# Review length categories
df['review_length'] = df['review'].apply(lambda x: len(x.split()))
df['review_type'] = pd.cut(df['review_length'],
                           bins=[0, 50, 150, float('inf')],
                           labels=['Short', 'Medium', 'Long'])

# Count values for each category
counts = df['review_type'].value_counts().reindex(['Short', 'Medium', 'Long'])

# Bar chart with border
plt.figure(figsize=(8,6))
bars = plt.bar(counts.index, counts.values, 
               color=['red', 'yellow', 'green'], 
               edgecolor='black', linewidth=1.5)

plt.title('Distribution of Short, Medium, and Long Reviews')
plt.xlabel('Review Type')
plt.ylabel('Count')
plt.show()
