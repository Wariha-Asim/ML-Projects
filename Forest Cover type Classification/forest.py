import streamlit as st
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
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\AR FAST\Documents\python mini projects\Elevvo Ml tasks\covtype.data", header=None)
    columns = [
        "Elevation", "Compass_Aspect", "Land_Slope",
        "Distance_To_Hydrology_Horizontal", "Distance_To_Hydrology_Vertical",
        "Distance_To_Roadways", "Sunlight_Morning_9AM", 
        "Sunlight_Noon_12PM", "Sunlight_Afternoon_3PM",
        "Distance_To_Fire_Points"
    ] + [f"Wilderness_Area_{i}" for i in range(1, 5)] \
      + [f"Soil_Category_{i}" for i in range(1, 41)] \
      + ["Forest_Cover_Type"]
    df.columns = columns
    return df

df = load_data()
st.title("🌲 Forest Cover Type Prediction Dashboard")
st.write("Original Dataset Shape:", df.shape)

# ================================
# Step 2: Use FULL Dataset but optimize model
# ================================
X = df.drop("Forest_Cover_Type", axis=1)
y = df["Forest_Cover_Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# Sidebar - User Input
# ================================
st.sidebar.header("🔹 Input Features")
input_data = {}


for col in df.columns[:10]:
    input_data[col] = st.sidebar.number_input(
        col, float(df[col].min()), float(df[col].max()), float(df[col].mean())
    )


for col in df.columns[10:-1]:   
    input_data[col] = 0

# final dataframe banado (same columns as training data)
user_input = pd.DataFrame([input_data], columns=X.columns)

# ================================
# Step 3: Model Selection
# ================================
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Decision Tree"])

# Optimized Models
if model_choice == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=10,     
        max_depth=12,        
        max_features="sqrt", 
        min_samples_split=10, 
        n_jobs=-1,           
        random_state=42
    )
else:
    model = DecisionTreeClassifier(
        max_depth=20,      # limited depth
        random_state=42
    )

# Train once
model.fit(X_train, y_train)
st.subheader(f"Currently using: {model_choice}")
y_pred = model.predict(X_test)

# ================================
# Step 4: Prediction
# ================================
if st.sidebar.button("Predict"):
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]
    st.success(f"🌳 Predicted Forest Cover Type for your input: **{prediction}**")

    # ================================
    # Step 5: Evaluation Metrics
    # ================================
    st.subheader("📊 Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
    st.write("Recall:", recall_score(y_test, y_pred, average="weighted"))
    st.write("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

    # ================================
    # Step 6: Confusion Matrix
    # ================================
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, None]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm_normalized, cmap="coolwarm", aspect="auto")  # changed to coolwarm
    ax.set_title("Confusion Matrix (Normalized)")
    plt.colorbar(im)

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")

    # Add numbers inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}",
                    ha="center", va="center", color="black")

    st.pyplot(fig)

    # ================================
    # Step 7: Feature Importances
    # ================================
    if model_choice in ["Decision Tree", "Random Forest"]:
        st.subheader("🌟 Feature Importances")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(range(len(indices)), importances[indices], align="center")
        ax2.set_xticks(range(len(indices)))
        ax2.set_xticklabels(X.columns[indices], rotation=90)
        ax2.set_title("Top 15 Feature Importances")
        st.pyplot(fig2)

    # ================================
    # Step 8: Classification Report
    # ================================
    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))
