import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform

# ================================
# 1. Load Dataset
# ================================
df = pd.read_csv('spam_detection_dataset.csv')
print(df.head(5))
print(df.isnull().sum())

# ================================
# 2. Feature Selection (Independent and Dependent Variables)
# ================================
X = df[['num_links', 'num_words', 'has_offer', 'sender_score', 'all_caps']]
y = df['is_spam']

# ================================
# 3. Split Dataset into Train and Test Sets
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test.shape)

# ================================
# 4. Feature Scaling (Standardization)
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# 5. Hyperparameter Tuning using RandomizedSearchCV
# ================================
param_dist = {
    'C': uniform(0.1, 10),        # Regularization parameter
    'gamma': ['scale', 'auto'],   # Kernel coefficient
    'kernel': ['linear', 'rbf']   # Try both linear and rbf
}

random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)
print("Best Parameters from RandomizedSearchCV:", random_search.best_params_)

# Final model with best parameters
model = random_search.best_estimator_

# ================================
# 6. Predictions on Test Data
# ================================
y_pred = model.predict(X_test_scaled)

print("====================================================")
print("Email Spam Classification Prediction Using SVM: ", y_pred[:10])

# ================================
# 7. Model Evaluation
# ================================
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📑 Classification Report:\n", classification_report(y_test, y_pred))

# ================================
# 8. Confusion Matrix
# ================================
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix: \n", cm)

# ================================
# 9. Confusion Matrix Plot
# ================================
plt.imshow(cm, cmap="PuRd")
plt.title("Confusion Matrix for Spam Detection")
plt.ylabel("Actual [Not Spam, Spam]")
plt.xlabel("Predicted [Not Spam, Spam]")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.colorbar()
plt.show()
