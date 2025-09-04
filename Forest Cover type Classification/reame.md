# 🌲 Forest Cover Type Prediction

This project predicts the **forest cover type** using machine learning models.  

It contains **two versions**:

1. **Streamlit Dashboard** → `forest_streamlit.py` (interactive web app)  
2. **Console / Python Script** → `forest_cover.py` (runs in terminal with model metrics and plots)  

Both versions use the same ML logic, but the Streamlit version provides a more interactive and user-friendly experience.

---

## 🔹 Features
- Data loading & preprocessing  
- Train-test split  
- Models used: **Decision Tree** & **Random Forest**  
- Model evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- Feature importance visualization  
- Streamlit version: real-time predictions and interactive plots  

---

## 📂 Dataset
- File: `covtype.data`  
- Recommended source: [Kaggle Forest Cover Type Dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)  
- Place the dataset in the same folder as the scripts  

---

## ⚙️ Steps & Features
1. Data Loading & Exploration  
2. Data Preprocessing  
3. Train-Test Split  
4. Model Training → Decision Tree & Random Forest  
5. Model Evaluation → Metrics & Confusion Matrix  
6. Feature Importance Visualization  
7. Streamlit Dashboard (interactive predictions & visualizations)  

---

## 📊 Visualizations
- Confusion Matrix  
- Feature Importance  
- Model Comparison  
- Streamlit dashboard for interactive exploration  

---

## 🧾 Results
- Models trained on forest cover dataset  
- Random Forest generally gives higher accuracy  
- Feature importance highlights key predictors for forest type  

---

## 🛠️ Technologies Used
- Python 3.x  
- Pandas → Data manipulation  
- Matplotlib → Visualization  
- Scikit-learn → Decision Tree, Random Forest, StandardScaler  
- Streamlit → Interactive dashboard (`forest_streamlit.py`)  

---

## 🚀 How to Run

### 1️⃣ Streamlit Version (`forest_streamlit.py`)
git clone https://github.com/Wariha-Asim/ML-Projects.git
cd "ML-Projects/Forest Cover Prediction"
pip install pandas matplotlib scikit-learn streamlit
streamlit run forest_streamlit.py

## Console / Python Script Version (forest_cover.py)
git clone https://github.com/Wariha-Asim/ML-Projects.git
cd "ML-Projects/Forest Cover Prediction"
pip install pandas matplotlib scikit-learn
python forest_cover.py

Runs in terminal / console and displays metrics, confusion matrix, and feature importance plots.
