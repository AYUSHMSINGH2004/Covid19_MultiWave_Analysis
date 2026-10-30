"""
Module 5: Clustering Algorithms
=================================
Syllabus Topics:
  - Introduction to Spectral Clustering
  - Document Clustering (applied to countries as "documents" with feature vectors)
  - Minimum Spanning Tree Clustering
  - Model-based Clustering (Expectation-Maximization / GMM)
  - Hierarchical Agglomerative Model-based Clustering
  - Outlier Detection using Clustering
"""

import os, sys, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                               davies_bouldin_score)
from sklearn.neighbors import NearestNeighbors

from utils.data_loader import build_country_features, PLOTS_DIR

print("=" * 80)
print("MODULE 5: CLUSTERING ALGORITHMS")
print("=" * 80)

# ── Data Preparation ─────────────────────────────────────────────────────────
features = build_country_features()
# Filter countries with at least 50,000 total cases for meaningful clustering
features_filtered = features[features["total_cases"] > 50000].copy()
print(f"  Countries with > 50K cases: {len(features_filtered)}")

feature_cols = ["total_cases", "total_deaths", "peak_daily_cases",
                "peak_daily_deaths", "mean_daily_cases", "mean_daily_deaths",
                "std_daily_cases", "std_daily_deaths", "cfr_percent", "case_volatility"]

X = features_filtered[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
country_labels = features_filtered["country"].values

# PCA for 2D visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
print(f"  PCA 2D explained variance: {pca_2d.explained_variance_ratio_.sum():.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.1  SPECTRAL CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5.1 SPECTRAL CLUSTERING")
print("─" * 60)

# Try different numbers of clusters
best_sil_spectral = -1
best_k_spectral = 2
for k in range(2, 8):
    sc = SpectralClustering(n_clusters=k, affinity="rbf", gamma=1.0,
                              random_state=42, n_init=10)
    labels = sc.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  k={k}: Silhouette = {sil:.4f}")
    if sil > best_sil_spectral:
        best_sil_spectral = sil
        best_k_spectral = k

sc_final = SpectralClustering(n_clusters=best_k_spectral, affinity="rbf",
                                gamma=1.0, random_state=42, n_init=10)
spectral_labels = sc_final.fit_predict(X_scaled)
print(f"\n  Best k = {best_k_spectral}, Silhouette = {best_sil_spectral:.4f}")

# Print cluster assignments
for c in range(best_k_spectral):
    members = country_labels[spectral_labels == c]
    print(f"  Cluster {c}: {len(members)} countries — {', '.join(members[:8])}"
          f"{'...' if len(members) > 8 else ''}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.2  MINIMUM SPANNING TREE (MST) CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5.2 MINIMUM SPANNING TREE (MST) CLUSTERING")
print("─" * 60)

# Build distance matrix and compute MST
dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
mst = minimum_spanning_tree(dist_matrix).toarray()
mst_symmetric = mst + mst.T

# Get MST edge weights
edges = []
for i in range(len(mst)):
    for j in range(i+1, len(mst)):
        if mst[i, j] > 0:
            edges.append((i, j, mst[i, j]))
        elif mst[j, i] > 0:
            edges.append((i, j, mst[j, i]))

edges.sort(key=lambda x: x[2], reverse=True)
print(f"  MST edges: {len(edges)}")
print(f"  Longest edges (candidates for cutting):")
for i, (u, v, w) in enumerate(edges[:5]):
    print(f"    Edge {i+1}: {country_labels[u]} ↔ {country_labels[v]}  (dist = {w:.3f})")

# Cut the top k-1 longest edges to form k clusters
n_clusters_mst = best_k_spectral
from scipy.sparse.csgraph import connected_components

mst_cut = mst_symmetric.copy()
for u, v, w in edges[:n_clusters_mst - 1]:
    mst_cut[u, v] = 0
    mst_cut[v, u] = 0

n_comp, mst_labels = connected_components(mst_cut, directed=False)
mst_sil = silhouette_score(X_scaled, mst_labels)
print(f"\n  After cutting {n_clusters_mst-1} longest edges: {n_comp} clusters")
print(f"  MST Clustering Silhouette Score: {mst_sil:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.3  MODEL-BASED CLUSTERING (EXPECTATION-MAXIMIZATION / GMM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5.3 MODEL-BASED CLUSTERING (GMM / EM ALGORITHM)")
print("─" * 60)

# Select optimal number of components using BIC and AIC
bic_scores = []
aic_scores = []
k_range = range(2, 8)
for k in k_range:
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=42, n_init=5)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))
    labels = gmm.predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  k={k}: BIC = {bic_scores[-1]:>10.1f}, AIC = {aic_scores[-1]:>10.1f}, "
          f"Silhouette = {sil:.4f}")

best_k_gmm = list(k_range)[np.argmin(bic_scores)]
print(f"\n  Optimal k (lowest BIC) = {best_k_gmm}")

gmm_final = GaussianMixture(n_components=best_k_gmm, covariance_type="full",
                              random_state=42, n_init=5)
gmm_labels = gmm_final.fit_predict(X_scaled)
gmm_probs = gmm_final.predict_proba(X_scaled)
gmm_sil = silhouette_score(X_scaled, gmm_labels)
print(f"  GMM Silhouette Score: {gmm_sil:.4f}")

# EM Algorithm details
print(f"\n  EM Convergence Details:")
print(f"    Iterations: {gmm_final.n_iter_}")
print(f"    Converged:  {gmm_final.converged_}")
print(f"    Log-Likelihood: {gmm_final.score(X_scaled) * len(X_scaled):,.2f}")

# Show countries with highest uncertainty (max posterior probability < 0.7)
max_probs = gmm_probs.max(axis=1)
uncertain_mask = max_probs < 0.7
if uncertain_mask.sum() > 0:
    print(f"\n  Uncertain Assignments (max probability < 70%):")
    for idx in np.where(uncertain_mask)[0][:10]:
        prob_str = ", ".join([f"C{j}:{p:.1%}" for j, p in enumerate(gmm_probs[idx])])
        print(f"    {country_labels[idx]}: [{prob_str}]")

# ─────────────────────────────────────────────────────────────────────────────
# 5.4  HIERARCHICAL AGGLOMERATIVE CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5.4 HIERARCHICAL AGGLOMERATIVE CLUSTERING")
print("─" * 60)

# Compare different linkage methods
for method in ["ward", "complete", "average", "single"]:
    Z = linkage(X_scaled, method=method, metric="euclidean")
    labels = fcluster(Z, t=best_k_spectral, criterion="maxclust")
    sil = silhouette_score(X_scaled, labels)
    print(f"  {method:>8s} linkage (k={best_k_spectral}): Silhouette = {sil:.4f}")

# Use Ward linkage (best for balanced clusters)
Z_ward = linkage(X_scaled, method="ward", metric="euclidean")
hier_labels = fcluster(Z_ward, t=best_k_spectral, criterion="maxclust") - 1
hier_sil = silhouette_score(X_scaled, hier_labels)

# Agglomerative Clustering via sklearn
agg = AgglomerativeClustering(n_clusters=best_k_spectral, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)
agg_sil = silhouette_score(X_scaled, agg_labels)
print(f"\n  AgglomerativeClustering (sklearn, Ward): Silhouette = {agg_sil:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.5  OUTLIER DETECTION USING CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("5.5 OUTLIER DETECTION USING CLUSTERING")
print("─" * 60)

# 5.5.1 DBSCAN-based outlier detection
# Find optimal eps using k-distance graph
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

dbscan = DBSCAN(eps=2.5, min_samples=3)
db_labels = dbscan.fit_predict(X_scaled)
n_outliers_db = (db_labels == -1).sum()
n_clusters_db = len(set(db_labels) - {-1})
print(f"  DBSCAN (eps=2.5, min_samples=3):")
print(f"    Clusters found: {n_clusters_db}")
print(f"    Outliers detected: {n_outliers_db}")

outlier_countries = country_labels[db_labels == -1]
if len(outlier_countries) > 0:
    print(f"    Outlier countries: {', '.join(outlier_countries)}")

# 5.5.2 GMM-based outlier detection (low posterior probability)
gmm_outlier_scores = -gmm_final.score_samples(X_scaled)
outlier_threshold = np.percentile(gmm_outlier_scores, 95)
gmm_outliers = gmm_outlier_scores > outlier_threshold
print(f"\n  GMM-based Outlier Detection (top 5% lowest log-likelihood):")
print(f"    Threshold: {outlier_threshold:.2f}")
print(f"    Outliers: {country_labels[gmm_outliers].tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("CLUSTERING EVALUATION METRICS SUMMARY")
print("─" * 60)

results = {
    "Method": ["Spectral", "MST", "GMM (EM)", "Hierarchical (Ward)", "Agglomerative"],
    "n_clusters": [best_k_spectral, n_clusters_mst, best_k_gmm, best_k_spectral, best_k_spectral],
    "Silhouette": [best_sil_spectral, mst_sil, gmm_sil, hier_sil, agg_sil],
}
all_labels_list = [spectral_labels, mst_labels, gmm_labels, hier_labels, agg_labels]

ch_scores = []
db_scores = []
for lbl in all_labels_list:
    n_unique = len(set(lbl))
    if n_unique > 1:
        ch_scores.append(calinski_harabasz_score(X_scaled, lbl))
        db_scores.append(davies_bouldin_score(X_scaled, lbl))
    else:
        ch_scores.append(0)
        db_scores.append(float("inf"))

results["Calinski-Harabasz"] = ch_scores
results["Davies-Bouldin"] = db_scores

eval_df = pd.DataFrame(results)
print(eval_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Figure 1: All clustering results in PCA space
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("Module 5 — Clustering Algorithms (PCA 2D Projection)",
             fontsize=15, fontweight="bold")

clustering_methods = {
    "Spectral Clustering": spectral_labels,
    "MST Clustering": mst_labels,
    "GMM (EM Algorithm)": gmm_labels,
    "Hierarchical (Ward)": hier_labels,
    "DBSCAN (Outlier Detection)": db_labels,
}

cmap = plt.cm.Set2
for ax, (method_name, labels) in zip(axes.flatten(), clustering_methods.items()):
    unique_labels = set(labels)
    for label in sorted(unique_labels):
        mask = labels == label
        color = "black" if label == -1 else cmap(label / max(max(unique_labels), 1))
        marker = "x" if label == -1 else "o"
        name = "Outlier" if label == -1 else f"Cluster {label}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], marker=marker,
                    s=40, label=name, alpha=0.7, edgecolors="black", linewidths=0.3)

    # Annotate notable countries
    for i, country in enumerate(country_labels):
        if country in ["US", "India", "Brazil", "Russia", "United Kingdom"]:
            ax.annotate(country, (X_2d[i, 0], X_2d[i, 1]), fontsize=6,
                          alpha=0.8, ha="left")

    ax.set_title(method_name, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

# Evaluation metrics bar chart in the 6th subplot
ax_eval = axes[1, 2]
x_pos = np.arange(len(eval_df))
bars = ax_eval.barh(eval_df["Method"], eval_df["Silhouette"],
                      color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"],
                      edgecolor="black", linewidth=0.3)
ax_eval.set_title("Silhouette Score Comparison", fontweight="bold")
ax_eval.set_xlabel("Silhouette Score")
for bar, val in zip(bars, eval_df["Silhouette"]):
    ax_eval.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                  f"{val:.3f}", va="center", fontsize=9)
ax_eval.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m5_clustering_results.png"), dpi=200)
plt.close()
print(f"\n  ✓ Saved: plots/m5_clustering_results.png")

# Figure 2: Dendrogram
fig, ax = plt.subplots(figsize=(18, 8))
# Truncate for readability
dn = dendrogram(Z_ward, labels=country_labels, leaf_rotation=90,
                 leaf_font_size=6, color_threshold=Z_ward[-best_k_spectral+1, 2],
                 above_threshold_color="gray", ax=ax)
ax.set_title("Module 5 — Hierarchical Agglomerative Clustering Dendrogram (Ward Linkage)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Distance (Ward)")
ax.axhline(Z_ward[-best_k_spectral+1, 2], color="#dc2626", linestyle="--",
            label=f"Cut at k={best_k_spectral}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m5_dendrogram.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m5_dendrogram.png")

# Figure 3: GMM BIC/AIC selection + K-distance plot for DBSCAN
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Module 5 — Model Selection & Outlier Detection Diagnostics",
             fontsize=14, fontweight="bold")

axes[0].plot(list(k_range), bic_scores, "o-", color="#2563eb", linewidth=2, label="BIC")
axes[0].plot(list(k_range), aic_scores, "s--", color="#dc2626", linewidth=2, label="AIC")
axes[0].axvline(best_k_gmm, color="#16a34a", linestyle=":", label=f"Optimal k = {best_k_gmm}")
axes[0].set_xlabel("Number of Components (k)")
axes[0].set_ylabel("Information Criterion")
axes[0].set_title("GMM: BIC & AIC vs k")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_distances, color="#2563eb", linewidth=1.5)
axes[1].axhline(2.5, color="#dc2626", linestyle="--", label="eps = 2.5")
axes[1].set_xlabel("Points (sorted by 5-NN distance)")
axes[1].set_ylabel("5th Nearest Neighbor Distance")
axes[1].set_title("K-Distance Graph (DBSCAN eps selection)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m5_model_selection.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m5_model_selection.png")

print("\n" + "=" * 80)
print("MODULE 5 COMPLETE")
print("=" * 80)
