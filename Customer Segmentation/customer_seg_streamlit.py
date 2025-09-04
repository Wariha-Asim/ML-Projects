import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="centered", page_icon="📊")
st.title("Customer Segmentation Dashboard")

st.subheader("Dataset Preview")
df = pd.read_csv(r"C:\Users\AR FAST\Documents\python mini projects\Elevvo Ml tasks\Mall_Customers.csv")
st.dataframe(df.head())

features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("Silhouette Scores")
sil_scores = {}
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=0)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores[k] = score

sil_df = pd.DataFrame({'Clusters': list(sil_scores.keys()), 'Silhouette Score': list(sil_scores.values())})
st.bar_chart(sil_df.set_index('Clusters'))

st.subheader("K-Means Clustering")
optimal_k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)

if st.button("Run K-Means"):
    model = KMeans(n_clusters=optimal_k, random_state=0)
    df['Cluster'] = model.fit_predict(X_scaled)
    centers = model.cluster_centers_

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', optimal_k)
    for i in range(optimal_k):
        cluster_data = df[df['Cluster'] == i]
        ax.scatter(cluster_data['Annual Income (k$)'],
                   cluster_data['Spending Score (1-100)'],
                   color=colors(i),
                   alpha=0.7,
                   label=f"Cluster {i}",
                   s=15)
        ax.text(centers[i,0]*scaler.scale_[0]+scaler.mean_[0],
                centers[i,1]*scaler.scale_[1]+scaler.mean_[1],
                f"C{i}", fontsize=8, fontweight='bold', color='black')

    ax.set_title("Income vs Spending Score")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(fontsize=5, loc='best')
    plt.tight_layout()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)

    st.subheader("Cluster Analysis")
    avg_income_all = df['Annual Income (k$)'].mean()
    avg_spending_all = df['Spending Score (1-100)'].mean()

    for i in range(optimal_k):
        cluster_data = df[df['Cluster'] == i]
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()

        if avg_income > avg_income_all and avg_spending > avg_spending_all:
            label = "Premium Customer"
        elif avg_income < avg_income_all and avg_spending < avg_spending_all:
            label = "Budget Customer"
        else:
            label = "Average Customer"

        st.write(f"**Cluster {i} ({label}):**")
        st.write(f"  - Average Income: {avg_income:.2f} k$")
        st.write(f"  - Income Range: {cluster_data['Annual Income (k$)'].min()} - {cluster_data['Annual Income (k$)'].max()} k$")
        st.write(f"  - Average Spending Score: {avg_spending:.2f}")
        st.write(f"  - Spending Score Range: {cluster_data['Spending Score (1-100)'].min()} - {cluster_data['Spending Score (1-100)'].max()}")
        st.write("---")
