# Task 2: Customer Segmentation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\AR FAST\Documents\python mini projects\Elevvo Ml tasks\Mall_Customers.csv")
print("Dataset Loaded Successfully. First 5 rows:")
print(df.head())

# Select features → Annual Income and Spending Score
annual_income = df['Annual Income (k$)']
spending_score = df['Spending Score (1-100)']

# Initial scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(annual_income, spending_score, marker='o', color='green', alpha=0.6)
plt.title("Annual Income vs Spending Score (Before Clustering)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Scale features → Normalize the values
scaled_cols = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler() 
x_scaled = scaler.fit_transform(scaled_cols)
print("\nFeatures scaled using StandardScaler.")

# Determine optimal number of clusters using Silhouette Score
print("\nSilhouette Scores for different cluster counts:")
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=0)
    labels = model.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)
    print(f"  Clusters={k}, Silhouette Score={score:.3f}")

# Train K-Means with optimal clusters
optimal_k = 3
model = KMeans(n_clusters=optimal_k, random_state=0)
model.fit(x_scaled)
result = model.predict(x_scaled)

centers = model.cluster_centers_
print(f"\nK-Means trained with {optimal_k} clusters.")
print("Cluster centers (scaled):")
print(centers)

# Scatter plot after clustering
plt.figure(figsize=(8, 5))
colors = plt.cm.get_cmap('tab10', optimal_k)
for i in range(optimal_k):
    plt.scatter(df[result==i]['Annual Income (k$)'],
                df[result==i]['Spending Score (1-100)'],
                color=colors(i),
                alpha=0.6,
                label=f"Cluster {i}")
    # Label at cluster center
    plt.text(centers[i,0]*scaler.scale_[0]+scaler.mean_[0],
             centers[i,1]*scaler.scale_[1]+scaler.mean_[1],
             f"C{i}",
             fontsize=12,
             fontweight='bold',
             color='black')

plt.title("Income vs Spending Score After Clustering (K-Means)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Analyze clusters → Average income and spending score per cluster
print("\nCluster Analysis:")
for i in range(optimal_k):
    cluster_data = df[result == i]
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()
    
    # Human-readable label
    if avg_income > df['Annual Income (k$)'].mean() and avg_spending > df['Spending Score (1-100)'].mean():
        label = "Premium Customer"
    elif avg_income < df['Annual Income (k$)'].mean() and avg_spending < df['Spending Score (1-100)'].mean():
        label = "Budget Customer"
    else:
        label = "Average Customer"
    
    print(f"\nCluster {i} ({label}):")
    print(f"  Average Income: {avg_income:.2f} k$")
    print(f"  Income Range: {cluster_data['Annual Income (k$)'].min()} - {cluster_data['Annual Income (k$)'].max()} k$")
    print(f"  Average Spending Score: {avg_spending:.2f}")
    print(f"  Spending Score Range: {cluster_data['Spending Score (1-100)'].min()} - {cluster_data['Spending Score (1-100)'].max()}")
