# Phishing Website Detection using SVM, Random Forest, and XGBoost with Hyperparameter Tuning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform

# Load dataset
df = pd.read_csv("dataset.csv")
print(df.head())
print(df.isnull().sum())
print(df.shape)
print(df.columns)

# Drop unnecessary column
df = df.drop(columns=['index'])

# Features and target
X = df.drop(columns=['Result'])
y = df['Result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline SVM
svm_baseline = SVC()
svm_baseline.fit(X_scaled, y_train)
svm_baseline_pred = svm_baseline.predict(X_test_scaled)
svm_baseline_pred = [0 if p == -1 else 1 for p in svm_baseline_pred]
print("  Accuracy (Baseline SVM):", accuracy_score(y_test.replace(-1, 0), svm_baseline_pred))

# RandomizedSearchCV on SVM
svm_param_dist = {
    "C": uniform(0.1, 10),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}
svm_random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=svm_param_dist,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=-1
)
y_train_svm = y_train.replace(-1, 0)
y_test_svm = y_test.replace(-1, 0)
svm_random_search.fit(X_scaled, y_train_svm)
print("Best Parameters (RandomizedSearchCV SVM):", svm_random_search.best_params_)
svm_model = svm_random_search.best_estimator_
svm_predict = svm_model.predict(X_test_scaled)
acc = accuracy_score(y_test_svm, svm_predict)
print("  Accuracy (Tuned SVM):", acc)

# GridSearchCV based on RandomizedSearchCV best params
g_svm_param_dist = {'C': [8.0, 8.4, 8.8], 'gamma': ['auto'], 'kernel': ['rbf']}
g_svm_random_search = GridSearchCV(SVC(), param_grid=g_svm_param_dist, cv=3, n_jobs=-1)
y_train_dt_g = y_train.replace(-1, 0)
y_test_dt_g = y_test.replace(-1, 0)
g_svm_random_search.fit(X_scaled, y_train_dt_g)
print("Best Parameters (GridSearchCV SVM):", g_svm_random_search.best_params_)
svm_g_model = g_svm_random_search.best_estimator_
svm_g_predict = svm_g_model.predict(X_test_scaled)
print("Accuracy (GridSearchCV SVM):", accuracy_score(y_test_dt_g, svm_g_predict))

# Compare GridSearchCV vs RandomizedSearchCV
acc_grid = accuracy_score(y_test_dt_g, svm_g_predict)
acc_random = accuracy_score(y_test_svm, svm_predict)
if acc_grid >= acc_random:
    print("Final Model: GridSearchCV SVM")
else:
    print("Final Model: RandomizedSearchCV SVM")

# Random Forest baseline
y_train_rf = y_train.replace(-1, 0)
y_test_rf = y_test.replace(-1, 0)
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_scaled, y_train_rf)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
rf_acc = accuracy_score(y_test_rf, y_pred_rf)
rf_cr = classification_report(y_test_rf, y_pred_rf)
rf_cm = confusion_matrix(y_test_rf, y_pred_rf)
rf_roc = roc_auc_score(y_test_rf, y_pred_proba_rf)
print("Accuracy (Random Forest):", rf_acc)

# XGBoost with RandomizedSearchCV
xgb_model = XGBClassifier(random_state=42)
xgb_param_dist = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1, 5]
}
xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_dist,
    cv=5,
    n_iter=10,
    n_jobs=-1,
    random_state=42
)
y_train_xgb = y_train.replace(-1, 0)
y_test_xgb = y_test.replace(-1, 0)
xgb_random_search.fit(X_scaled, y_train_xgb)
print("Best Parameters (XGBoost):", xgb_random_search.best_params_)
best_xgb_model = xgb_random_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test_xgb, y_pred_xgb)
xgb_cm = confusion_matrix(y_test_xgb, y_pred_xgb)
print("Accuracy (XGBoost):", xgb_acc)

# Confusion Matrices
rf_cm_percent = rf_cm.astype("float") / rf_cm.sum() * 100
svm_cm = confusion_matrix(y_test_svm, svm_predict)
svm_cm_percent = svm_cm.astype("float") / svm_cm.sum() * 100
xgb_cm_percent = xgb_cm.astype("float") / xgb_cm.sum() * 100

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
im1 = axes[0].imshow(rf_cm_percent, interpolation="nearest", cmap="PuRd")
axes[0].set_title("Random Forest - Confusion Matrix (%)")
for i in range(rf_cm_percent.shape[0]):
    for j in range(rf_cm_percent.shape[1]):
        axes[0].text(j, i, f"{rf_cm_percent[i, j]:.1f}%", ha="center", va="center")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(svm_cm_percent, interpolation="nearest", cmap="Blues")
axes[1].set_title("SVM (Tuned) - Confusion Matrix (%)")
for i in range(svm_cm_percent.shape[0]):
    for j in range(svm_cm_percent.shape[1]):
        axes[1].text(j, i, f"{svm_cm_percent[i, j]:.1f}%", ha="center", va="center")
fig.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(xgb_cm_percent, interpolation="nearest", cmap="Greens")
axes[2].set_title("XGBoost (Tuned) - Confusion Matrix (%)")
for i in range(xgb_cm_percent.shape[0]):
    for j in range(xgb_cm_percent.shape[1]):
        axes[2].text(j, i, f"{xgb_cm_percent[i, j]:.1f}%", ha="center", va="center")
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()

# Bar Chart Comparison (SVM vs RF)
svm_report = classification_report(y_test_svm, svm_predict, output_dict=True)
svm_precision = svm_report['weighted avg']['precision']
svm_recall = svm_report['weighted avg']['recall']
svm_f1 = svm_report['weighted avg']['f1-score']
svm_roc = roc_auc_score(y_test_svm, svm_predict)

rf_precision = float(rf_cr.split()[9])
rf_recall = float(rf_cr.split()[10])
rf_f1 = float(rf_cr.split()[11])

metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
svm_scores = [acc, svm_precision, svm_recall, svm_f1, svm_roc]
rf_scores = [rf_acc, rf_precision, rf_recall, rf_f1, rf_roc]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, svm_scores, width, label="SVM (Tuned)", color='purple')
bars2 = ax.bar(x + width/2, rf_scores, width, label="Random Forest (Baseline)", color='pink')
ax.set_ylabel("Score")
ax.set_title("Phishing Website Detection: Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Save Random Forest predictions
submission = pd.DataFrame({"id": X_test.index, "prediction": y_pred_rf})
submission.to_csv("rf_submission.csv", index=False)
print("Submission CSV created: rf_submission.csv")
