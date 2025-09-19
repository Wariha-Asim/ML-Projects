# 📰 Fake News Classification Project  

## 📌 Project Overview  
The rapid spread of misinformation has become one of the biggest challenges in the digital age.  
This project aims to **classify news articles as either Fake (0) or Real (1)** using machine learning models.  
We combine **data preprocessing, feature extraction, and multiple ML algorithms** to identify the most effective model for detecting fake news.  


## 📊 Dataset  
We used two publicly available datasets:  

- **Fake.csv** → Contains fake news articles.  
- **True.csv** → Contains real news articles.  

Both datasets were merged into a single dataframe with a new **label column**:  
- `0` → Fake News  
- `1` → Real News  

### Preprocessing steps applied:  
- Merging `title` and `text` columns  
- Converting all text to lowercase  
- Removing punctuation, numbers, and symbols  
- Lemmatization using **WordNet Lemmatizer**  
- Stopword removal with **NLTK**  


## ⚙️ Project Features  
✔️ **Text Preprocessing:** Cleaning, tokenization, lemmatization, stopword removal  
✔️ **Feature Extraction:** Bag-of-Words using `CountVectorizer`  
✔️ **Model Training & Comparison:**  
   - **Naive Bayes**  
   - **Logistic Regression**  
   - **Random Forest**  
✔️ **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  
✔️ **Visualizations:** Word distributions, histograms, and word clouds for Fake vs Real news  



## 🏆 Results & Best Model  

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | **0.98** | **0.98**  | **0.98** | **0.98** | **0.98** |
| Random Forest       | 0.97     | 0.97      | 0.97   | 0.97     | 0.97    |
| Naive Bayes         | 0.94     | 0.95      | 0.92   | 0.93     | 0.94    |

✅ **Best Model:**  
**Logistic Regression** delivered the highest accuracy (≈98%), precision, recall, and ROC-AUC, making it the most reliable choice for Fake News Detection.  


## 📂 Project Structure  
Fake-News-Classification-Project/
│
├──fake_news.py 
├── README.md # Project documentation


## 📑 Submission.csv  
Predictions were generated using the **Logistic Regression model** (98% accuracy).  
This file contains the model’s outputs for further analysis.  


## 📌 Kaggle Notebook  
You can explore the full notebook on Kaggle here:  
👉 [Fake News Classification Project](https://www.kaggle.com/code/waarihaasim/fake-news-classification-project)  


