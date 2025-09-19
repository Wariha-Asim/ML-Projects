# Phishing Website Detection - Machine Learning Project

## Overview
This project implements a complete **machine learning pipeline** for detecting phishing websites using multiple classification algorithms. It covers all steps from **data preprocessing** to **model evaluation and comparison**.


## Key Features

### 🔍 Data Preprocessing
- Dataset with **31 features** and **11,055 entries**
- Missing values handled and statistical analysis performed
- Feature scaling using **StandardScaler**
- Train-test split with **80-20 ratio**

### 🤖 Machine Learning Models
1. **Support Vector Machine (SVM)**  
   - Baseline and tuned implementations  
   - Accuracy range: **91.7% – 93.7%**

2. **Random Forest Classifier**  
   - Implemented with 200 estimators  
   - Achieved **95.7% accuracy**  
   - **ROC AUC score: 0.991**  
   - Best-performing model  

3. **XGBoost Classifier**  
   - Hyperparameter tuning via RandomizedSearchCV  
   - Comparative analysis with other models  

### 📊 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix analysis  
- ROC Curve and AUC Score  


## Technical Highlights
- **Data**: 32 features including URL characteristics, domain details, and web traffic metrics  
- **Preprocessing**: Scaling with StandardScaler  
- **Validation**: 5-fold cross-validation for robust results

### Project Struture

Phishing-Website-Detection/
│
├── phishing_detection.py # Main script
├── README.md # Project overview & results
└── rf_submission.csv # Random Forest predictions (CSV)


## Results
The **Random Forest Classifier** delivered the best performance:  
- ✅ **95.7% Accuracy**  
- 📊 **0.991 ROC AUC Score**  
- ⚡ Excellent precision and recall balance  

> A predictions file (`rf_submission.csv`) is included for reference and further analysis.  


## Kaggle Notebook
You can view the full notebook with code and visualizations here:  
👉 [Phishing Website Detection - Kaggle Notebook](https://www.kaggle.com/code/waarihaasim/phishing-detection-website-notebook)


## Project Structure
