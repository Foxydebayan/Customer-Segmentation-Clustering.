import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# Load Dataset
data = pd.read_csv('Mall_Customers.csv')
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ================================
# 1️⃣ Elbow Method
# ================================
wcss = []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_pca)
    wcss.append(km.inertia_)

plt.figure()
plt.plot(K, wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.show()

# ================================
# 2️⃣ Silhouette Score
# ================================
sil_scores = []
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_pca)
    sil_scores.append(silhouette_score(X_pca, labels))

plt.figure()
plt.plot(K, sil_scores, marker='o')
plt.title("Silhouette Analysis")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# Choose k (from plots, usually 5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
k_labels = kmeans.fit_predict(X_pca)

# ================================
# 3️⃣ K-Means Cluster Plot
# ================================
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='*', label='Centroids')
plt.title(f"K-Means Clustering (k={optimal_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# ================================
# 4️⃣ DBSCAN (Outlier Detection)
# ================================
dbscan = DBSCAN(eps=0.6, min_samples=5)
db_labels = dbscan.fit_predict(X_pca)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap='plasma')
plt.title("DBSCAN Clustering (Outliers = -1)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# ================================
# 5️⃣ Distance Metric Comparison
# ================================
euclid_mean = np.mean(euclidean_distances(X_pca))
manhat_mean = np.mean(manhattan_distances(X_pca))

print("Average Euclidean Distance:", euclid_mean)
print("Average Manhattan Distance:", manhat_mean)

# ================================
# 6️⃣ Dendrogram
# ================================
linked = linkage(X_pca, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()