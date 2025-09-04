import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================================
# Step 1: Load Dataset
# ================================
df = pd.read_csv(r"C:\Users\AR FAST\Documents\python mini projects\Elevvo Ml tasks\covtype.data", header=None)

# Assign column names
columns = [
    "Elevation", 
    "Compass_Aspect", 
    "Land_Slope",
    "Distance_To_Hydrology_Horizontal",  
    "Distance_To_Hydrology_Vertical",
    "Distance_To_Roadways",
    "Sunlight_Morning_9AM", 
    "Sunlight_Noon_12PM", 
    "Sunlight_Afternoon_3PM",
    "Distance_To_Fire_Points"
] + [f"Wilderness_Area_{i}" for i in range(1, 5)] \
  + [f"Soil_Category_{i}" for i in range(1, 41)] \
  + ["Forest_Cover_Type"]

df.columns = columns

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print("First 5 rows:\n", df.head(5), "\n")

# ================================
# Step 2: Split Features and Target
# ================================
X = df[[ 
    "Elevation", "Compass_Aspect", "Land_Slope",
    "Distance_To_Hydrology_Horizontal", "Distance_To_Hydrology_Vertical",
    "Distance_To_Roadways", "Sunlight_Morning_9AM", 
    "Sunlight_Noon_12PM", "Sunlight_Afternoon_3PM",
    "Distance_To_Fire_Points"
] + [f"Wilderness_Area_{i}" for i in range(1, 5)] \
  + [f"Soil_Category_{i}" for i in range(1, 41)]]

y = df["Forest_Cover_Type"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# Step 3: Train Random Forest
# ================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Model Performance")
print("------------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred_rf, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred_rf, average='weighted'), "\n")

# ================================
# Step 4: Train Decision Tree
# ================================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Model Performance")
print("----------------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred_dt, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred_dt, average='weighted'), "\n")

# ================================
# Step 5: Model Comparison
# ================================
results = pd.DataFrame({
    "Model": ["Random Forest", "Decision Tree"],
    "Accuracy": [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_dt)],
    "Precision": [precision_score(y_test, y_pred_rf, average='weighted'), precision_score(y_test, y_pred_dt, average='weighted')],
    "Recall": [recall_score(y_test, y_pred_rf, average='weighted'), recall_score(y_test, y_pred_dt, average='weighted')],
    "F1-Score": [f1_score(y_test, y_pred_rf, average='weighted'), f1_score(y_test, y_pred_dt, average='weighted')]
})

print(" Model Comparison:\n", results.to_string(index=False), "\n")

# Bar chart comparison
results.set_index("Model")[["Accuracy","Precision","Recall","F1-Score"]].plot(kind="bar", figsize=(8,6))
plt.title("Random Forest vs Decision Tree Performance")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0) 
plt.show()

# ================================
# Step 6: Confusion Matrix (Decision Tree)
# ================================
cm = confusion_matrix(y_test, y_pred_dt)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, None]

plt.figure(figsize=(8,6))
im = plt.imshow(cm_normalized, interpolation="nearest", cmap="coolwarm")
plt.title("Confusion Matrix (Decision Tree - Normalized)")
plt.colorbar(im)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")


for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                 ha="center", va="center", color="black")

plt.show()

# ================================
# Step 7: Feature Importances (Decision Tree)
# ================================
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10,6))
plt.bar(range(len(indices)), importances[indices], color="green",align="center")
plt.xticks(range(len(indices)), X.columns[indices],rotation=90)
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Decision Tree)")
plt.ylabel("Importance Score")
plt.show()



# ================================
# Step 8: Classification Report
# ================================
print("Classification Report (Decision Tree):\n")
print(classification_report(y_test, y_pred_dt))
