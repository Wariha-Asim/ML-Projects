## 🛍 Customer Segmentation using K-Means Clustering

This project applies K-Means clustering to segment mall customers based on their Annual Income and Spending Score. By identifying customer groups, businesses can design better marketing strategies and improve decision-making.

## 📂 Dataset

The dataset used is Mall_Customers.csv, which contains the following columns:

CustomerID → Unique customer identifier

Gender → Male / Female

Age → Age of the customer

Annual Income (k$) → Annual income in thousands of dollars

Spending Score (1-100) → Spending behavior score assigned by the mall

## ⚙️ Features:
1. Data Loading & Exploration

Loaded the dataset using Pandas

Previewed data with .head() to understand structure

2. Feature Selection

Selected Annual Income and Spending Score for clustering

3. Data Visualization (Before Clustering)

Scatter plot of income vs. spending score

4. Feature Scaling

Standardized features using StandardScaler for better clustering performance

5. Optimal Cluster Selection

Used Silhouette Score to test clusters k=2 to 10

Selected k=3 as the optimal number of clusters

6. K-Means Clustering

Trained K-Means with optimal k=3

Visualized results with distinct cluster colors and labeled cluster centers

7. Cluster Analysis

For each cluster:

Calculated average income and average spending score

Categorized clusters as:

Premium Customers → High income, high spending

Budget Customers → Low income, low spending

Average Customers → Middle income/spending

## 📊 Visualizations

Scatter Plot (Before Clustering) → Shows raw data distribution

Scatter Plot (After Clustering) → Shows segmented groups with cluster centers

## 🧾 Results

Optimal number of clusters: 3

Identified distinct customer groups with meaningful insights

Example outputs:

Premium Customers → Target with luxury services

Budget Customers → Offer discounts & budget deals

Average Customers → Retain with balanced offers

## 🛠️ Technologies Used

Python 3.x

Pandas → Data manipulation

Matplotlib → Data visualization

Scikit-learn → K-Means clustering, scaling, silhouette score

## 🚀 How to Run

Clone this repository:

git clone https://github.com/Wariha-Asim/Customer Segmentation.git
cd customerseg.py


## Install dependencies:

pip install pandas matplotlib scikit-learn


Run the script:

python customer_segmentation.py
