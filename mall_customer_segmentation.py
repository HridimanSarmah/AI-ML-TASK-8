import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
n_samples = 200
data = {
    "CustomerID": range(1, n_samples + 1),
    "Gender": np.random.choice(["Male", "Female"], n_samples),
    "Age": np.random.randint(18, 70, n_samples),
    "Annual Income (k$)": np.random.randint(15, 135, n_samples),
    "Spending Score (1-100)": np.random.randint(1, 101, n_samples)
}
df = pd.DataFrame(data)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

X = df.drop(columns=["CustomerID"])
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Final cluster visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_pca["Cluster"] = df["Cluster"]

sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Cluster", palette="Set2")
plt.title("Customer Segments")
plt.show()
