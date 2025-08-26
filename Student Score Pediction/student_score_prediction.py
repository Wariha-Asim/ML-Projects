import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# ======================
#  Load Dataset
# ======================
df = pd.read_csv(r"C:\Users\AR FAST\Documents\python mini projects\Elevvo Ml tasks\StudentPerformanceFactors.csv")

#check inconsistent columns
print("Columns with Null values")
print(df.isnull().sum())

# Fill missing values (using mode for categorical/ordinal columns)
df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])
print("Modified Dataframe: ")
print(df.head())
# ======================
# Data Visualization
# ======================
# Histogram of Key Features
ploting1 = df[['Hours_Studied', 'Previous_Scores', 'Sleep_Hours', 'Exam_Score']]
plt.figure(figsize=(10,6))
colors=['red', 'green', 'blue', 'orange']
plt.hist(ploting1, bins=10, color=colors, edgecolor='black', alpha=0.5)
plt.title('Histogram Comparison of Key Features')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Scatter Plot: Hours Studied vs Exam Score
plt.figure(figsize=(10,6))
plt.scatter(df['Hours_Studied'], df['Exam_Score'], color='purple')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score')
plt.show()

# Attendance vs Average Exam Score
avg_scores = df.groupby('Attendance')['Exam_Score'].mean()
plt.figure(figsize=(8,6))
avg_scores.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Attendance vs Average Exam Score")
plt.xlabel("Attendance Level")
plt.ylabel("Average Exam Score")
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()

# ======================
# Train-Test Split
# ======================
X = df[['Hours_Studied','Previous_Scores','Sleep_Hours']]
y = df['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# Linear Regression
# ======================
model = LinearRegression()
model.fit(X_train, y_train)
predicted_scores = model.predict(X_test)

# Evaluation Metrics for Linear Regression
mae = mean_absolute_error(y_test, predicted_scores)
mse = mean_squared_error(y_test, predicted_scores)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicted_scores)

# Residual Plot (Linear Regression)
residuals = y_test - predicted_scores
plt.figure(figsize=(8,6))
plt.scatter(predicted_scores, residuals, color="green", alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Scores")
plt.ylabel("Residuals")
plt.title("Residual Plot (Linear Regression)")
plt.show()


print("\n===============================")
print(" Linear Regression Results")
print("===============================")
print("Predicted Exam Scores (first 10):")
print(predicted_scores[:10])
print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# ======================
# Polynomial Regression
# ======================
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluation Metrics for Polynomial Regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\n===============================")
print(" Polynomial Regression Results (Degree 2)")
print("===============================")
print("Predicted Exam Scores (first 10):")
print(y_pred_poly[:10])
print("\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse_poly:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_poly:.2f}")
print(f"R² Score: {r2_poly:.2f}")

# ======================
#  Comparison Table
# ======================
results = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression (Degree 2)"],
    "MSE": [mse, mse_poly],
    "RMSE": [rmse, rmse_poly],
    "R² Score": [r2, r2_poly]
})

print("\n===============================")
print(" Model Performance Comparison")
print("===============================")
print(results.to_string(index=False))

# Actual vs Predicted Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, predicted_scores, color='blue', alpha=0.6, label="Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2, label="Perfect Fit")
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores (Linear Regression)")
plt.legend()
plt.show()


# Actual vs Predicted Plot (Polynomial)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_poly, color='green', alpha=0.6, label="Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Perfect Fit")
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores (Polynomial)")
plt.title("Actual vs Predicted Exam Scores (Polynomial Regression)")
plt.legend()
plt.show()

