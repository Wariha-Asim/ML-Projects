import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv(r"C:\Users\AR FAST\Documents\python mini projects\Elevvo Ml tasks\loan_approval_dataset.csv")

# Check for missing values
print(df.isnull().sum())

# Encode categorical features
df.columns = df.columns.str.strip()  
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])
df['loan_status'] = le.fit_transform(df['loan_status'])

print("Encoded Data")
print(df.head())

# Feature & target split
X = df[['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 
        'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]
y = df['loan_status']

# Standard scaling
scaler = StandardScaler() 
x_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
loan_status = model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

#printing predictions
print("\n===============================")
print("Logistic Regression Predictions (first 10):")
print("===============================")
print(loan_status[:10])

print("\n===============================")
print("Decision Tree Model Predictions (first 10):")
print("===============================")
print(y_pred_dt[:10])

# Model performance
print("\nModel Performance (Logistic Regression):")
print("Accuracy:", accuracy_score(y_test, loan_status))
print("Precision:", precision_score(y_test, loan_status))
print("Recall:", recall_score(y_test, loan_status))
print("F1-Score:", f1_score(y_test, loan_status))

print("\nModel Performance (Decision Tree):")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1-Score:", f1_score(y_test, y_pred_dt))

# Model comparison
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [accuracy_score(y_test, loan_status), accuracy_score(y_test, y_pred_dt)],
    "Precision": [precision_score(y_test, loan_status), precision_score(y_test, y_pred_dt)],
    "Recall": [recall_score(y_test, loan_status), recall_score(y_test, y_pred_dt)],
    "F1-Score": [f1_score(y_test, loan_status), f1_score(y_test, y_pred_dt)]
})

print("\nModel Comparison:")
print(results.to_string(index=False))

results.set_index("Model")[["Accuracy","Precision","Recall","F1-Score"]].plot(kind="bar", figsize=(8,6))
plt.title("Logistic Regression vs Decision Tree")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0) 
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, loan_status)
print("\nConfusion Matrix:")
print(cm)

# Loan status distribution
plt.title("Loan Status Distribution")
plt.pie(df['loan_status'].value_counts(),
        autopct='%1.1f%%',
        labels=['Approved', 'Not Approved'],
        shadow=True,
        startangle=360)
plt.show()

# Loan status vs education
edu_counts = df.groupby(['loan_status', 'education']).size().unstack(fill_value=0)
print(edu_counts)

plt.figure(figsize=(8,6))
plt.bar(edu_counts.index, edu_counts[1], label="Graduate", color="skyblue")
plt.bar(edu_counts.index, edu_counts[0], bottom=edu_counts[1], label="Not Graduate", color="salmon")
plt.title("Loan Status vs Education")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.legend(title="Education")
plt.show()
