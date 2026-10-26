"""Prompt 2: K-Means (Elbow -> PCA scatter -> summary table)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES = ["price_usd", "cushioning_score", "energy_return_pct", "weight_g"]
df = pd.read_csv("running_shoes.csv")
X = StandardScaler().fit_transform(df[FEATURES])
sns.set_theme(style="whitegrid", context="talk")

# 1) Elbow method
ks = np.arange(1, 11)
inertia = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]

# elbow = point farthest from the line joining first and last inertia values
p1, p2 = np.array([ks[0], inertia[0]]), np.array([ks[-1], inertia[-1]])
d = [abs((p2[0]-p1[0])*(p1[1]-i) - (p1[0]-k)*(p2[1]-p1[1])) / np.linalg.norm(p2 - p1)
     for k, i in zip(ks, inertia)]
best_k = int(ks[int(np.argmax(d))])
print(f"Optimal K (elbow): {best_k}")

plt.figure(figsize=(9, 6))
plt.plot(ks, inertia, "o-", lw=2.5)
plt.axvline(best_k, color="crimson", ls="--", label=f"Elbow K = {best_k}")
plt.title("Elbow Method", weight="bold"); plt.xlabel("Number of clusters (K)"); plt.ylabel("Inertia")
plt.legend(); plt.tight_layout(); plt.savefig("02a_elbow.png", dpi=200); plt.show()

# 2) Final model + PCA plot
df["cluster"] = KMeans(best_k, n_init=10, random_state=42).fit_predict(X)
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X)
df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="cluster", palette="Set2",
                s=120, edgecolor="white")
plt.title("K-Means Clusters (PCA projection)", weight="bold")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
plt.tight_layout(); plt.savefig("02b_pca_clusters.png", dpi=200); plt.show()

# 3) Cluster summary table
summary = df.groupby("cluster")[FEATURES].mean().round(2)
summary.insert(0, "n_shoes", df["cluster"].value_counts().sort_index())
print("\nCluster summary:\n", summary)
summary.to_csv("02c_cluster_summary.csv")
