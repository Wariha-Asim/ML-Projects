# 🛍 Customer Segmentation using K-Means Clustering

This project applies **K-Means clustering** to segment mall customers based on their **Annual Income** and **Spending Score**.  
It contains **two versions**:

1. **Streamlit Dashboard** → `customer_seg_streamlit.py` (interactive web app)  
2. **Console / Python Script** → `customer_seg.py` (runs in terminal with matplotlib plots)  

Segmentation helps businesses design better marketing strategies and target customers effectively.

---

## 🔹 Features

- Segment customers into clusters based on income and spending behavior
- Determine the **optimal number of clusters** using Silhouette Score
- Visualize clusters and their centers
- Analyze each cluster:
  - Average income
  - Average spending score
  - Human-readable labels: Premium, Budget, or Average Customers

---

## 📂 Dataset

- Dataset file: `Mall_Customers.csv`
- Columns:
  - `CustomerID` → Unique customer identifier
  - `Gender` → Male / Female
  - `Age` → Customer age
  - `Annual Income (k$)` → Annual income in thousands of dollars
  - `Spending Score (1-100)` → Spending behavior score assigned by the mall

---

## ⚙️ Steps & Features

1. **Data Loading & Exploration**
2. **Feature Selection** → `Annual Income` and `Spending Score`
3. **Data Visualization (Before Clustering)** → Scatter plot
4. **Feature Scaling** → StandardScaler
5. **Optimal Cluster Selection** → Silhouette Score for `k=2` to `10`
6. **K-Means Clustering** → Train with optimal `k` (default 3)
7. **Cluster Analysis** → Categorize clusters: Premium, Budget, Average

---

## 📊 Visualizations

- Scatter Plot Before Clustering  
- Scatter Plot After Clustering  
- Silhouette Score Bar Chart (Streamlit version only)  

---

## 🧾 Results

- Optimal clusters: **3**
- Cluster insights:
  - **Premium Customers** → High income, high spending
  - **Budget Customers** → Low income, low spending
  - **Average Customers** → Middle income/spending

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas → Data manipulation
- Matplotlib → Visualization
- Scikit-learn → K-Means clustering, StandardScaler, Silhouette Score
- Streamlit → Interactive dashboard (`customer_seg_streamlit.py`)

---

## 🚀 How to Run

### 1️⃣ Streamlit Version (`customer_seg_streamlit.py`)


git clone https://github.com/Wariha-Asim/ML-Projects.git
cd "ML-Projects/Customer Segmentation"

## Console / Python Script Version (customer_seg.py)
git clone https://github.com/Wariha-Asim/ML-Projects.git
cd "ML-Projects/Customer Segmentation"
pip install pandas matplotlib scikit-learn
python customer_seg.py

## Runs in terminal / console

Displays cluster analysis and shows scatter plots before and after clustering

pip install pandas matplotlib scikit-learn streamlit
streamlit run customer_seg_streamlit.py
