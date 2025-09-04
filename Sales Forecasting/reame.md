
# 📈 Sales Forecasting Dashboard

A **Python-based machine learning project** for predicting weekly sales of stores and departments using historical sales data. Supports multiple regression models and visualizations to compare predictions with actual sales.

---

## 🔹 Features

- Predict weekly sales using:
  - Linear Regression
  - Random Forest Regressor
- Time-series-aware train-test split
- Feature engineering:
  - Date decomposition (day, month, year)
  - Lag features (previous week sales)
  - Encoding categorical features
- Standardized numeric features
- Evaluation metrics: MAE, MSE, RMSE, R² Score
- Visualizations:
  - Model performance comparison
  - Monthly average sales vs predicted sales

---

## 📂 Dataset

- Dataset file: `train.csv`  
- Columns include:
  - `Store` – store identifier
  - `Dept` – department identifier
  - `Date` – date of sales
  - `Weekly_Sales` – sales value (target)
  - `IsHoliday` – boolean indicator for holiday weeks

---

## ⚡ Installation

git clone https://github.com/<Wariha-Asim>/Sales forecasting.git
cd Sales-forecasting.py

## 🚀 Usage

Ensure train.csv is in the correct path.

Run the script.

The following will be generated:

Model evaluation metrics table

Bar charts comparing MAE, MSE, RMSE, and R² for Linear Regression and Random Forest

Line chart showing monthly average sales: Actual vs Predicted

## 🧠 Models

Linear Regression – baseline regression model

Random Forest Regressor – ensemble model for improved accuracy

## 📊 Visualizations

Metrics Comparison: MAE, MSE, RMSE, R² score per model

Monthly Average Sales: Actual vs Predicted trends
cd sales-forecasting
pip install -r requirements.txt
