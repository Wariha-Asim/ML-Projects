# 🎬 Sentiment Analysis of Movie Reviews

## 📌 Project Overview
This project applies **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to classify IMDB movie reviews as **Positive (1)** or **Negative (0)**.  

The pipeline includes **data preprocessing, feature extraction, model training, performance evaluation, and visualization** to identify the best-performing sentiment classification model.

- **Kaggle Notebook:** My full Step by Step Notebook: [Sentiment Analysis on IMDB Movie Reviews](https://www.kaggle.com/code/waarihaasim/sentiment-analysis-on-imdb-movie-reviews)

---

## 📊 Dataset
- **Source:** IMDB Movie Review Dataset  
- **Size:** 50,000 labeled reviews  
- **Features:**  
  - `review`: text content of movie reviews  
  - `sentiment`: target label (0 = Negative, 1 = Positive)  

---

## ⚙️ Project Features

### 🔍 Data Preprocessing
- Removed **HTML tags, punctuation, numbers, and special characters**  
- Converted text to **lowercase**  
- Applied **tokenization, stopword removal, and lemmatization**  
- Balanced and shuffled dataset for fair training  

### 🧩 Feature Extraction
- **CountVectorizer** used for **Naive Bayes** & **Logistic Regression**  
- **TF-IDF Vectorization** used for **Support Vector Machine (SVM)**  

### 🤖 Machine Learning Models
1. **Naïve Bayes (MultinomialNB)**  
   - Quick baseline model with moderate accuracy  

2. **Logistic Regression**  
   - Strong linear model  
   - Balanced precision and recall  

3. **Support Vector Machine (SVM)**  
   - Implemented with TF-IDF features  
   - Trained on a **sample of 10,000 reviews**  
   - Best overall performance  

---

## 📊 Model Evaluation
- Metrics: **Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC**  
- Side-by-side comparison of all models  
- Overall ranking based on **Accuracy, F1-Score, ROC-AUC**  

---

## 📈 Visualizations
- **Confusion Matrix Heatmap** (for Logistic Regression)  
- **Word Clouds** for Positive & Negative reviews  
- **Histogram of Review Length Distribution** (Positive vs Negative)  
- **Bar Chart of Review Length Categories** (Short, Medium, Long)  

---
## Project Structure
🎬 Sentiment Analysis on IMDB Movie Reviews/
│
├── sentiment_analysis.py
├── readme.md


## 🏆 Results & Best Model
- **Naïve Bayes** → Good baseline, fast but less accurate  
- **Logistic Regression** → Reliable, balanced performance  
- **SVM** → Highest overall performance on sampled data  

✅ **Selected Best Model: Support Vector Machine (SVM)**  
- Accuracy: ~**86%**  
- Precision/Recall: High & consistent  
- ROC-AUC: Strong classification power  

---

## 📌 Conclusion
This project demonstrates the power of **text preprocessing + ML models** in sentiment classification.  
The **SVM model with TF-IDF features** proved to be the best choice for analyzing IMDB reviews on the sampled dataset, achieving robust evaluation metrics.

---

## 📂 Links
- **Kaggle Notebook:** [Sentiment Analysis on IMDB Movie Reviews](https://www.kaggle.com/code/waarihaasim/sentiment-analysis-on-imdb-movie-reviews)
