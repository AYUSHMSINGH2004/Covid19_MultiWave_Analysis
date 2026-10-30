"""
Module 6: Dimensionality Reduction
====================================
Syllabus Topics:
  Linear Methods:
    - Principal Component Analysis (PCA)
    - Singular Value Decomposition (SVD)
    - Factor Analysis
    - Intrinsic Dimensionality
  Non-Linear Methods:
    - Multidimensional Scaling (MDS)
    - Manifold Learning (t-SNE)
    - Self-Organizing Maps (SOM)
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import svd as scipy_svd

from utils.data_loader import build_country_features, PLOTS_DIR

print("=" * 80)
print("MODULE 6: DIMENSIONALITY REDUCTION")
print("=" * 80)

# ── Data Preparation ─────────────────────────────────────────────────────────
features = build_country_features()
features_filtered = features[features["total_cases"] > 50000].copy()

feature_cols = ["total_cases", "total_deaths", "peak_daily_cases",
                "peak_daily_deaths", "mean_daily_cases", "mean_daily_deaths",
                "std_daily_cases", "std_daily_deaths", "cfr_percent", "case_volatility"]

X = features_filtered[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
country_labels = features_filtered["country"].values
n_samples, n_features = X_scaled.shape
print(f"  Data: {n_samples} countries × {n_features} features")

# ─────────────────────────────────────────────────────────────────────────────
# 6.1  PRINCIPAL COMPONENT ANALYSIS (PCA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6.1 PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("─" * 60)

pca_full = PCA()
X_pca = pca_full.fit_transform(X_scaled)

print(f"\n  Explained Variance Ratio per Component:")
cumulative = 0
for i, (ev, evr) in enumerate(zip(pca_full.explained_variance_,
                                    pca_full.explained_variance_ratio_)):
    cumulative += evr
    print(f"    PC{i+1}: Variance = {ev:.4f}, Ratio = {evr:.4f} "
          f"({evr*100:.1f}%), Cumulative = {cumulative*100:.1f}%")

# How many PCs needed for 95% variance
n_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
print(f"\n  Components for 95% variance: {n_95}")
print(f"  Components for 90% variance: "
      f"{np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.90) + 1}")

# PCA loadings (feature contributions)
loadings = pd.DataFrame(
    pca_full.components_[:4].T,
    index=feature_cols,
    columns=[f"PC{i+1}" for i in range(4)]
)
print(f"\n  PCA Loadings (top 4 components):")
print(loadings.round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6.2  SINGULAR VALUE DECOMPOSITION (SVD)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6.2 SINGULAR VALUE DECOMPOSITION (SVD)")
print("─" * 60)

U, S, Vt = scipy_svd(X_scaled, full_matrices=False)

print(f"  U shape:  {U.shape}  (left singular vectors)")
print(f"  S shape:  {S.shape}  (singular values)")
print(f"  Vt shape: {Vt.shape} (right singular vectors)")

# Singular values and their explained variance
total_var = np.sum(S**2)
svd_var_ratio = (S**2) / total_var
print(f"\n  Singular Values and Variance Explained:")
cum_svd = 0
for i in range(min(10, len(S))):
    cum_svd += svd_var_ratio[i]
    print(f"    σ{i+1} = {S[i]:.4f}, Variance = {svd_var_ratio[i]*100:.1f}%, "
          f"Cumulative = {cum_svd*100:.1f}%")

# Reconstruct with k components
for k in [2, 3, 5]:
    X_recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    recon_error = np.linalg.norm(X_scaled - X_recon, "fro")
    total_norm = np.linalg.norm(X_scaled, "fro")
    print(f"\n  Rank-{k} reconstruction: error = {recon_error:.4f} "
          f"({recon_error/total_norm*100:.1f}% of total)")

# ─────────────────────────────────────────────────────────────────────────────
# 6.3  FACTOR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6.3 FACTOR ANALYSIS")
print("─" * 60)

# Determine optimal number of factors
fa_scores = []
for n_factors in range(2, min(n_features, 8)):
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa.fit(X_scaled)
    ll = fa.score(X_scaled)
    fa_scores.append((n_factors, ll))
    print(f"  {n_factors} factors: Log-Likelihood = {ll:.4f}")

best_n_factors = max(fa_scores, key=lambda x: x[1])[0]
print(f"\n  Optimal factors (highest log-likelihood): {best_n_factors}")

fa_final = FactorAnalysis(n_components=best_n_factors, random_state=42)
X_fa = fa_final.fit_transform(X_scaled)

# Factor loadings
fa_loadings = pd.DataFrame(
    fa_final.components_.T,
    index=feature_cols,
    columns=[f"Factor {i+1}" for i in range(best_n_factors)]
)
print(f"\n  Factor Loadings:")
print(fa_loadings.round(3).to_string())

# Communalities (variance explained per feature)
communalities = 1 - fa_final.noise_variance_
comm_df = pd.DataFrame({
    "Feature": feature_cols,
    "Communality": communalities,
    "Specificity": fa_final.noise_variance_
})
print(f"\n  Communalities (variance captured by factors):")
print(comm_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 6.4  INTRINSIC DIMENSIONALITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6.4 INTRINSIC DIMENSIONALITY")
print("─" * 60)

# Method 1: PCA-based (elbow / 95% variance threshold)
pca_intrinsic_95 = n_95
print(f"  Method 1 — PCA (95% variance threshold):        d = {pca_intrinsic_95}")

pca_intrinsic_90 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.90) + 1
print(f"  Method 1 — PCA (90% variance threshold):        d = {pca_intrinsic_90}")

# Method 2: Kaiser Criterion (eigenvalues > 1)
eigenvalues = pca_full.explained_variance_
kaiser_d = np.sum(eigenvalues > 1.0)
print(f"  Method 2 — Kaiser Criterion (eigenvalues > 1):  d = {kaiser_d}")

# Method 3: Maximum Likelihood Estimation (MLE) of intrinsic dimensionality
def mle_intrinsic_dim(X, k=5):
    """Levina & Bickel (2005) MLE estimator for intrinsic dimensionality."""
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # exclude self

    # For each point, compute local dimensionality estimate
    local_dims = []
    for i in range(len(X)):
        d = distances[i]
        d = d[d > 0]  # avoid log(0)
        if len(d) < 2:
            continue
        Tk = d[-1]
        log_ratios = np.log(Tk / d[:-1])
        if log_ratios.sum() > 0:
            local_dim = (len(d) - 1) / log_ratios.sum()
            local_dims.append(local_dim)

    return np.mean(local_dims) if local_dims else 0

mle_d = mle_intrinsic_dim(X_scaled, k=5)
print(f"  Method 3 — MLE (Levina-Bickel, k=5):            d = {mle_d:.2f}")

# Method 4: Correlation Dimension (box-counting inspired)
print(f"\n  Summary: Intrinsic dimensionality ≈ {int(round(np.mean([pca_intrinsic_95, kaiser_d, mle_d])))} "
      f"(average of methods)")

# ─────────────────────────────────────────────────────────────────────────────
# 6.5  MULTIDIMENSIONAL SCALING (MDS)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6.5 MULTIDIMENSIONAL SCALING (MDS)")
print("─" * 60)

mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean",
           normalized_stress="auto", max_iter=500)
X_mds = mds.fit_transform(X_scaled)
stress = mds.stress_
print(f"  MDS Stress: {stress:.4f}")
print(f"  Interpretation: {'Good' if stress < 0.1 else 'Moderate' if stress < 0.2 else 'Poor'} "
      f"(< 0.1 is good)")

# ─────────────────────────────────────────────────────────────────────────────
# 6.6  MANIFOLD LEARNING (t-SNE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6.6 MANIFOLD LEARNING (t-SNE)")
print("─" * 60)

# Run t-SNE with different perplexities
perplexities = [5, 15, 30]
tsne_results = {}
for p in perplexities:
    try:
        tsne = TSNE(n_components=2, perplexity=p, random_state=42,
                    max_iter=1000, learning_rate="auto", init="pca")
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=p, random_state=42,
                    n_iter=1000, learning_rate="auto", init="pca")
    X_tsne = tsne.fit_transform(X_scaled)
    kl_div = tsne.kl_divergence_
    tsne_results[p] = (X_tsne, kl_div)
    print(f"  Perplexity = {p:>2d}: KL divergence = {kl_div:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.7  SELF-ORGANIZING MAPS (SOM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("6.7 SELF-ORGANIZING MAPS (SOM)")
print("─" * 60)

class SimpleSOM:
    """Minimal Self-Organizing Map implementation for demonstration."""

    def __init__(self, grid_x=5, grid_y=5, input_dim=10,
                 sigma=1.5, learning_rate=0.5, random_state=42):
        rng = np.random.RandomState(random_state)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.weights = rng.randn(grid_x, grid_y, input_dim)
        self.sigma_init = sigma
        self.lr_init = learning_rate

    def _find_bmu(self, x):
        """Find Best Matching Unit."""
        dists = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(dists), dists.shape)

    def _neighborhood(self, bmu, sigma):
        """Gaussian neighborhood function."""
        xx, yy = np.meshgrid(range(self.grid_x), range(self.grid_y), indexing="ij")
        d_sq = (xx - bmu[0])**2 + (yy - bmu[1])**2
        return np.exp(-d_sq / (2 * sigma**2))

    def fit(self, X, n_epochs=200):
        n = len(X)
        total_iter = n_epochs * n
        iteration = 0
        for epoch in range(n_epochs):
            indices = np.random.permutation(n)
            for i in indices:
                t = iteration / total_iter
                sigma = self.sigma_init * np.exp(-3 * t)
                lr = self.lr_init * np.exp(-3 * t)

                bmu = self._find_bmu(X[i])
                h = self._neighborhood(bmu, max(sigma, 0.01))
                self.weights += lr * h[:, :, np.newaxis] * (X[i] - self.weights)
                iteration += 1
        return self

    def predict(self, X):
        """Return BMU coordinates for each input."""
        bmus = []
        for x in X:
            bmu = self._find_bmu(x)
            bmus.append(bmu)
        return np.array(bmus)

    def quantization_error(self, X):
        """Average distance between inputs and their BMU weights."""
        errors = []
        for x in X:
            bmu = self._find_bmu(x)
            errors.append(np.linalg.norm(x - self.weights[bmu]))
        return np.mean(errors)

som = SimpleSOM(grid_x=6, grid_y=6, input_dim=n_features, sigma=2.0,
                learning_rate=0.5, random_state=42)
som.fit(X_scaled, n_epochs=200)
bmu_coords = som.predict(X_scaled)
qe = som.quantization_error(X_scaled)
print(f"  SOM Grid: 6 × 6 = 36 neurons")
print(f"  Quantization Error: {qe:.4f}")
print(f"  Training Epochs: 200")

# Count hits per neuron
hit_map = np.zeros((6, 6), dtype=int)
for bmu in bmu_coords:
    hit_map[bmu[0], bmu[1]] += 1

print(f"\n  SOM Hit Map (countries per neuron):")
print(hit_map)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Figure 1: PCA Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Module 6 — PCA & SVD Analysis", fontsize=14, fontweight="bold")

# Scree plot
axes[0, 0].bar(range(1, n_features+1), pca_full.explained_variance_ratio_,
                color="#60a5fa", edgecolor="black", linewidth=0.3, label="Individual")
axes[0, 0].plot(range(1, n_features+1), np.cumsum(pca_full.explained_variance_ratio_),
                "ro-", linewidth=2, label="Cumulative")
axes[0, 0].axhline(0.95, color="#16a34a", linestyle="--", alpha=0.7, label="95% threshold")
axes[0, 0].axhline(0.90, color="#f59e0b", linestyle="--", alpha=0.7, label="90% threshold")
axes[0, 0].set_xlabel("Principal Component")
axes[0, 0].set_ylabel("Explained Variance Ratio")
axes[0, 0].set_title("Scree Plot")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# PCA Biplot
axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], s=30, alpha=0.6, c="#3b82f6",
                     edgecolors="black", linewidths=0.3)
# Add loading vectors
for i, feat in enumerate(feature_cols):
    axes[0, 1].arrow(0, 0, pca_full.components_[0, i] * 5,
                       pca_full.components_[1, i] * 5,
                       color="#dc2626", alpha=0.8, head_width=0.15)
    axes[0, 1].text(pca_full.components_[0, i] * 5.5,
                      pca_full.components_[1, i] * 5.5,
                      feat.replace("_", "\n"), fontsize=5.5, color="#dc2626")

for i, country in enumerate(country_labels):
    if country in ["US", "India", "Brazil", "Russia", "United Kingdom", "France", "Germany"]:
        axes[0, 1].annotate(country, (X_pca[i, 0], X_pca[i, 1]), fontsize=6)
axes[0, 1].set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)")
axes[0, 1].set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)")
axes[0, 1].set_title("PCA Biplot (Data + Loadings)")
axes[0, 1].grid(True, alpha=0.3)

# SVD singular values
axes[1, 0].bar(range(1, len(S)+1), S, color="#f59e0b", edgecolor="black", linewidth=0.3)
axes[1, 0].set_xlabel("Component")
axes[1, 0].set_ylabel("Singular Value (σ)")
axes[1, 0].set_title("SVD: Singular Values")
axes[1, 0].grid(True, alpha=0.3)

# Kaiser criterion
axes[1, 1].bar(range(1, len(eigenvalues)+1), eigenvalues,
                color=["#22c55e" if e > 1 else "#e5e7eb" for e in eigenvalues],
                edgecolor="black", linewidth=0.3)
axes[1, 1].axhline(1.0, color="#dc2626", linestyle="--", label="Kaiser Threshold (λ = 1)")
axes[1, 1].set_xlabel("Component")
axes[1, 1].set_ylabel("Eigenvalue (λ)")
axes[1, 1].set_title(f"Kaiser Criterion → d = {kaiser_d}")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m6_pca_svd.png"), dpi=200)
plt.close()
print(f"\n  ✓ Saved: plots/m6_pca_svd.png")

# Figure 2: Factor Analysis loadings heatmap
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(fa_loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Loading"})
ax.set_title(f"Module 6 — Factor Analysis Loadings ({best_n_factors} Factors)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Original Feature")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m6_factor_analysis.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m6_factor_analysis.png")

# Figure 3: Non-linear methods
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Module 6 — Non-Linear Dimensionality Reduction",
             fontsize=14, fontweight="bold")

# MDS
axes[0, 0].scatter(X_mds[:, 0], X_mds[:, 1], s=40, c="#2563eb", alpha=0.7,
                     edgecolors="black", linewidths=0.3)
for i, country in enumerate(country_labels):
    if country in ["US", "India", "Brazil", "Russia", "United Kingdom", "France"]:
        axes[0, 0].annotate(country, (X_mds[i, 0], X_mds[i, 1]), fontsize=7)
axes[0, 0].set_title(f"MDS (Stress = {stress:.4f})", fontweight="bold")
axes[0, 0].set_xlabel("Dimension 1")
axes[0, 0].set_ylabel("Dimension 2")
axes[0, 0].grid(True, alpha=0.3)

# t-SNE at different perplexities
for idx, p in enumerate(perplexities):
    ax = axes[0, 1] if idx == 0 else axes[0, 2] if idx == 1 else axes[1, 0]
    X_tsne, kl_div = tsne_results[p]
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=40, c="#e11d48", alpha=0.7,
                edgecolors="black", linewidths=0.3)
    for i, country in enumerate(country_labels):
        if country in ["US", "India", "Brazil", "Russia", "United Kingdom"]:
            ax.annotate(country, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=7)
    ax.set_title(f"t-SNE (perplexity={p}, KL={kl_div:.3f})", fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

# SOM Hit Map
im = axes[1, 1].imshow(hit_map, cmap="YlOrRd", interpolation="nearest")
axes[1, 1].set_title("SOM Hit Map (6×6 grid)", fontweight="bold")
axes[1, 1].set_xlabel("SOM X")
axes[1, 1].set_ylabel("SOM Y")
for (j, i), val in np.ndenumerate(hit_map):
    axes[1, 1].text(i, j, str(val), ha="center", va="center",
                      fontsize=10, color="black" if val < hit_map.max()/2 else "white")
plt.colorbar(im, ax=axes[1, 1], label="# Countries")

# SOM U-Matrix (unified distance matrix)
u_matrix = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        neighbors = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < 6 and 0 <= nj < 6:
                neighbors.append(np.linalg.norm(
                    som.weights[i,j] - som.weights[ni,nj]))
        u_matrix[i, j] = np.mean(neighbors)

im2 = axes[1, 2].imshow(u_matrix, cmap="bone_r", interpolation="bilinear")
axes[1, 2].set_title("SOM U-Matrix (Cluster Boundaries)", fontweight="bold")
axes[1, 2].set_xlabel("SOM X")
axes[1, 2].set_ylabel("SOM Y")
plt.colorbar(im2, ax=axes[1, 2], label="Avg Neighbor Distance")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m6_nonlinear_methods.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m6_nonlinear_methods.png")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("DIMENSIONALITY REDUCTION SUMMARY")
print("─" * 60)
summary = pd.DataFrame({
    "Method": ["PCA", "SVD", "Factor Analysis", "MDS", "t-SNE (p=15)", "SOM (6×6)"],
    "Type": ["Linear", "Linear", "Linear", "Non-Linear", "Non-Linear", "Non-Linear"],
    "Output Dims": [n_95, n_95, best_n_factors, 2, 2, "6×6 grid"],
    "Key Metric": [
        f"95% var in {n_95} PCs",
        f"Fro. error {np.linalg.norm(X_scaled - U[:,:n_95]@np.diag(S[:n_95])@Vt[:n_95,:], 'fro'):.2f}",
        f"LL = {fa_final.score(X_scaled):.3f}",
        f"Stress = {stress:.4f}",
        f"KL = {tsne_results[15][1]:.4f}",
        f"QE = {qe:.4f}"
    ]
})
print(summary.to_string(index=False))

print("\n" + "=" * 80)
print("MODULE 6 COMPLETE")
print("=" * 80)
