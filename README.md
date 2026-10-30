# COVID-19 Multiwave Epidemiological Analysis & Machine Learning Project
## Complete Curriculum Implementation: Modules 2 through 7

This repository contains the end-to-end data science and machine learning pipeline for multi-wave COVID-19 pandemic analysis. It fulfills the entire course curriculum specified from **Module 2 to Module 7** in one unified, modular, reproducible framework.

---

## 1. Quick Start: Single-Stretch Execution

To run the entire curriculum from Module 2 to Module 7 in one automated stretch:

```bash
# Using the virtual environment:
.\venv\Scripts\python.exe run_all_modules.py

# Or if activated:
python run_all_modules.py
```

### Execution Output & Verification
Running `run_all_modules.py` sequentially executes each module, prints real-time analytical tables and mathematical metrics to stdout, and generates **23 high-resolution visualization artifacts** in the `plots/` directory:

```
================================================================================
COVID-19 MULTI-WAVE ANALYSIS -- COMPLETE MODULE EXECUTION (MODULES 2 -> 7)
================================================================================

>>> Running Module 2: Data Transformation [1/6]
  [PASSED] in 6.4s

>>> Running Module 3: Correlation & Time Series Analysis [2/6]
  [PASSED] in 17.9s

>>> Running Module 4: Data Summarization & Visualization [3/6]
  [PASSED] in 11.3s

>>> Running Module 5: Clustering Algorithms [4/6]
  [PASSED] in 9.2s

>>> Running Module 6: Dimensionality Reduction [5/6]
  [PASSED] in 9.2s

>>> Running Module 7: Model Development & Evaluation [6/6]
  [PASSED] in 9.1s

================================================================================
EXECUTION SUMMARY
================================================================================
  [PASSED]  Module 2: Data Transformation                      (6.4s)
  [PASSED]  Module 3: Correlation & Time Series Analysis       (17.9s)
  [PASSED]  Module 4: Data Summarization & Visualization       (11.3s)
  [PASSED]  Module 5: Clustering Algorithms                    (9.2s)
  [PASSED]  Module 6: Dimensionality Reduction                 (9.2s)
  [PASSED]  Module 7: Model Development & Evaluation           (9.1s)

  Total time: 63.2s
  Generated plots: 23 files in plots/
================================================================================
```

---

## 2. Project Architecture & File Organization

```
Covid-19 Multiwave Analysis/
│
├── run_all_modules.py                 # Master runner executing Modules 2-7 in one stretch
├── requirements.txt                   # Frozen python dependencies
├── .gitignore                         # Standard repository ignore rules
├── README.md                          # Master documentation & curriculum reference
│
├── modules/                           # Curriculum Modules 2 through 7
│   ├── __init__.py
│   ├── module2_data_transformation.py     # Module 2: Deduplication, Imputation, Binning, MLE
│   ├── module3_correlation_timeseries.py  # Module 3: Univariate, Bivariate, Multivariate & TSA
│   ├── module4_summarization_visualization.py # Module 4: 1D/2D/nD Summaries, Contingency, Plots
│   ├── module5_clustering.py              # Module 5: Spectral, MST, GMM, Agglomerative, DBSCAN
│   ├── module6_dimensionality_reduction.py # Module 6: PCA, SVD, Factor Analysis, MDS, t-SNE, SOM
│   └── module7_model_evaluation.py        # Module 7: Regressions, Residuals, CV, RL Q-Learning
│
├── utils/                             # Core shared utilities
│   ├── __init__.py
│   └── data_loader.py                 # Tidy time-series loading & country feature engineering
│
├── data/                              # JHU CSSE Raw Epidemic Datasets
│   ├── time_series_covid19_confirmed_global.csv
│   └── time_series_covid19_deaths_global.csv
│
├── dashboard/                         # MongoDB & Streamlit/Dash Interactive Analytics
│   ├── __init__.py
│   ├── app.py                         # Streamlit multi-wave interactive web app
│   ├── detect_waves.py                # Multi-wave peak detection using SciPy signal analysis
│   ├── ingest_data.py                 # MongoDB bulk dataset ingestion script
│   └── run_aggregation.py             # MongoDB aggregation pipeline for 7-day smoothing
│
└── plots/                             # Directory storing all 23 generated figures (.png)
    ├── legacy_waves/                  # Multi-wave epidemiological detection plots
    │   ├── covid_waves_Brazil.png
    │   ├── covid_waves_India.png
    │   ├── covid_waves_Italy.png
    │   ├── covid_waves_US.png
    │   └── covid_waves_United_Kingdom.png
    │
    ├── m2_binning_distributions.png
    ├── m2_missing_data_imputation.png
    ├── m3_univariate_analysis.png
    ├── m3_bivariate_analysis.png
    ├── m3_multivariate_correlation.png
    ├── m3_time_series_decomposition.png
    ├── m3_acf_pacf.png
    ├── m3_multi_country_timeseries.png
    ├── m3_grouping_resampling.png
    ├── m4_1d_analysis.png
    ├── m4_2d_analysis.png
    ├── m4_contingency_tables.png
    ├── m4_nd_analysis.png
    ├── m4_bar_dot_charts.png
    ├── m5_clustering_results.png
    ├── m5_dendrogram.png
    ├── m5_model_selection.png
    ├── m6_pca_svd.png
    ├── m6_factor_analysis.png
    ├── m6_nonlinear_methods.png
    ├── m7_regression_evaluation.png
    ├── m7_residual_diagnostics.png
    └── m7_learning_cv_rl.png
```

---

## 3. Detailed Curriculum & Syllabus Breakdown

### **Module 2: Data Transformation**
- **Data Deduplication:** Exact row duplicate detection (`df.duplicated()`) and logical domain duplicate checks across composite keys (`Country/Region` + `Date`). Includes injection and automated cleanup demonstration.
- **Replacing Values:** 
  - Zero-case early reporting handling.
  - Statistical outlier replacement ($> \mu + 3\sigma$) replaced with the median to preserve robust central tendency.
  - Categorical transformation mapping daily counts into epidemiological severity tiers (`Negligible`, `Low`, `Moderate`, `Severe`).
- **Discretization & Binning:**
  - Equal-Width Binning (`pd.cut` with 5 uniform intervals).
  - Equal-Frequency / Quantile Binning (`pd.qcut` with 4 quartiles: Q1, Q2, Q3, Q4).
  - Domain-Specific Binning (`Minimal`, `Controlled`, `Spreading`, `Surge`, `Crisis`).
- **Missing Data & Imputation:**
  - Traditional techniques: Mean imputation, Median imputation, Forward Fill (`ffill`), Backward Fill (`bfill`), Linear Time-Series Interpolation.
  - **Maximum Likelihood Estimation (MLE) via Expectation-Maximization (EM):** Iterative parameter estimation for missing values assuming normal observation distribution:
    $$\mu^{(t+1)} = \frac{1}{N}\left(\sum_{x_i \text{ obs}} x_i + \sum_{x_j \text{ miss}} \mu^{(t)}\right)$$
  - Quantitative benchmark comparison evaluating imputation Root Mean Squared Error (RMSE) against ground truth.

---

### **Module 3: Correlation Analysis and Time Series Analysis**
- **Types of Analysis:**
  - **Univariate Analysis:** Skewness, kurtosis, kernel density estimation (KDE), and boxplots.
  - **Bivariate Analysis:** Daily Cases vs. Daily Deaths evaluated via Pearson ($r = 0.697$), Spearman rank ($\rho = 0.771$), and Kendall's Tau ($\tau = 0.589$).
  - **Multivariate Analysis:** Cross-country correlation matrix across 10 major nations and pairwise joint distributions.
- **Time Series Analysis (TSA):**
  - **Characteristics & Decomposition:** Additive/multiplicative decomposition isolating Trend, 7-day Seasonality, and Residual components.
  - **Stationarity:** Rolling 30-day mean and variance tracking non-stationary pandemic waves.
  - **Autocorrelation (ACF & PACF):** Serial correlation diagnostics identifying strong lag-1 and lag-7 weekly cycle persistence.
  - **Time-based Indexing, Grouping & Resampling:** Monthly grouping, day-of-week reporting bias analysis, weekly (`W`) resampling, and 7-day rolling window smoothing.

---

### **Module 4: Data Summarization and Visualization**
- **Statistical Summary Measures:** Count, Mean, Median, Mode, Variance, Standard Deviation, Range, IQR, Skewness, Kurtosis, Coefficient of Variation, and Percentile Distribution ($P_1$ to $P_{99}$).
- **1-D Statistical Analysis:** Distribution histograms, mean/median vertical markers, side-by-side comparative boxplots, violin plots, and empirical Cumulative Distribution Functions (eCDF).
- **2-D Statistical Analysis:** Hexagonal binning density plots, 2D bivariate Gaussian KDE contours, and Quantile-Quantile (Q-Q) normal probability plots.
- **Contingency Tables & Independence Testing:**
  - Two-way contingency table cross-tabulating Case Severity vs. Death Severity.
  - Pearson's Chi-Square Test of Independence ($\chi^2 = 1,437.13, p = 7.34 \times 10^{-304}, \text{DoF} = 9$).
  - Effect size calculation using **Cramér's V** ($V = 0.6474$).
- **n-D Statistical Analysis:**
  - Parallel coordinates chart displaying normalized multidimensional country profiles.
  - 3-D Scatter plot ($X$: Cumulative Cases, $Y$: Cumulative Deaths, $Z$: Case Fatality Rate).
  - Spider / Radar chart comparing multidimensional wave severity signatures across global hotspots.
- **Visualizations:** Cleveland dot plots for Case Fatality Rates and grouped/horizontal bar charts for cumulative morbidity.

---

### **Module 5: Clustering Algorithms**
- **Spectral Clustering:** RBF similarity graph construction, Graph Laplacian computation, and normalized cut spectral embedding.
- **Document Clustering:** TF-IDF feature extraction on epidemic research abstracts with K-Means text clustering and silhouette evaluation.
- **Minimum Spanning Tree (MST) Clustering:** Full pairwise Euclidean distance graph calculation, Prim/Kruskal MST construction, and clustering via removal of the $k-1$ longest tree edges.
- **Model-Based Clustering:** Gaussian Mixture Models (GMM) optimized via the Expectation-Maximization (EM) algorithm. Model order selection evaluated using Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC).
- **Hierarchical Agglomerative Clustering:** Ward's minimum variance linkage, complete linkage, and dendrogram construction with Cophenetic Correlation Coefficient validation ($c = 0.814$).
- **Outlier Detection via Clustering:** Density-based anomaly detection using DBSCAN ($min\_samples=3, \epsilon=1.5$) and GMM lowest log-likelihood density thresholding.
- **Validation Metrics:** Silhouette Score ($S = 0.641$), Calinski-Harabasz Index ($CH = 141.2$), and Davies-Bouldin Index ($DB = 0.548$).

---

### **Module 6: Dimensionality Reduction**
- **Linear Methods:**
  - **Principal Component Analysis (PCA):** Full eigenvalue/eigenvector decomposition, scree plot analysis, cumulative explained variance ($58.3\%$ in PC1, $74.8\%$ in 2 PCs, $94.7\%$ in 4 PCs), and feature loading biplots.
  - **Singular Value Decomposition (SVD):** Full economy SVD ($X = U \Sigma V^T$) and low-rank matrix approximations (Rank-2, Rank-3, Rank-5) with Frobenius norm reconstruction error tracking ($12.1\%$ error at rank 5).
  - **Factor Analysis:** Maximum likelihood estimation of latent factor loadings, uniqueness/specificities, communalities, and model log-likelihood comparison across factor counts.
  - **Intrinsic Dimensionality Estimation:** Multi-method consensus estimation combining PCA 95% variance threshold ($d=5$), Kaiser Criterion (eigenvalues $> 1, d=3$), and Levina-Bickel Maximum Likelihood Estimation ($d=4.22$).
- **Non-Linear Methods:**
  - **Multidimensional Scaling (MDS):** Pairwise distance metric preservation evaluated with Kruskal's Stress formula.
  - **Manifold Learning (t-SNE):** Non-linear probability distribution mapping tested across multiple perplexities ($\text{Perp} = 5, 15, 30$) with Kullback-Leibler (KL) divergence minimization.
  - **Self-Organizing Maps (SOM):** $6 \times 6$ Kohonen artificial neural lattice with 2D Gaussian neighborhood adaptation, country hit-map activation frequencies, and Quantization Error computation ($QE = 0.4345$).

---

### **Module 7: Model Development and Evaluation**
- **Linear Regression Models:**
  - *Model 1 — Simple Linear Regression:* Daily Cases $\to$ Daily Deaths ($R^2 = 0.234, \text{RMSE} = 9,681$).
  - *Model 2 — Multiple Linear Regression:* Full multi-feature predictor matrix predicting country-level mortality.
  - *Model 3 — Autoregressive Distributed Lag (ARDL) Time-Series Model:* Incorporates 7-day, 14-day, and 21-day lagged case counts to capture incubation-to-mortality physiological delay ($R^2 = 0.735, \text{RMSE} = 385.68$).
- **Comprehensive Evaluation Metrics:**
  - Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Median Absolute Error, Mean Absolute Percentage Error (MAPE), Max Error.
  - Goodness-of-Fit: $R^2$, Adjusted $R^2$ (penalizing redundant parameters), Explained Variance Score.
  - In-depth analysis of accuracy concepts (regression metrics vs. classification accuracy, Bias-Variance tradeoff).
- **Residual Diagnostics & Assumption Testing:**
  - Residuals vs. Fitted plot (heteroscedasticity check).
  - Normal Q-Q plot of standardized residuals.
  - Residual autocorrelation plot & Durbin-Watson test.
  - Cook's Distance identification of high-leverage influential outliers.
- **Model Validation & Comparisons:**
  - 5-Fold Cross-Validation (mean $\pm$ std for $R^2$, MAE, RMSE).
  - Learning Curves (training vs. cross-validation performance across sample sizes).
  - Comparative benchmark: Ordinary Least Squares (OLS) vs. Ridge ($L_2$) vs. Lasso ($L_1$) vs. Polynomial Regression (degree 2).
- **Reinforcement Learning (RL) Framework:**
  - Theoretical comparison: Supervised Learning vs. Reinforcement Learning (feedback mechanism, sequential decision making, exploration vs. exploitation).
  - Working **Q-Learning Epidemic Simulation:** Markov Decision Process (MDP) modeling optimal pandemic intervention policies (balancing public health containment against economic lockdown penalties, $\epsilon$-greedy exploration, Q-table convergence).
  - Survey of real-world RL applications (AlphaGo, robotics, autonomous driving, healthcare dosing, dynamic epidemic management, RLHF for LLMs).

---

## 4. Complete List of Generated Visualizations

All figures are automatically saved to `plots/`:

| File Name | Module | Visualization Content |
|:---|:---:|:---|
| `m2_binning_distributions.png` | 2 | Equal-width, equal-frequency, and domain-specific bin distributions |
| `m2_missing_data_imputation.png` | 2 | Traditional methods vs. EM Maximum Likelihood Estimation comparison |
| `m3_univariate_analysis.png` | 3 | Histogram, KDE, and boxplot distribution analysis |
| `m3_bivariate_analysis.png` | 3 | Scatter plot with OLS trendline, Pearson/Spearman/Kendall correlation |
| `m3_multivariate_correlation.png` | 3 | Cross-country correlation heatmaps and pairwise joint distributions |
| `m3_time_series_decomposition.png` | 3 | Observed, Trend, 7-Day Seasonality, and Residual components |
| `m3_acf_pacf.png` | 3 | Autocorrelation (ACF) and Partial Autocorrelation (PACF) correlograms |
| `m3_multi_country_timeseries.png` | 3 | Multi-nation chronological multi-wave infection timelines |
| `m3_grouping_resampling.png` | 3 | Monthly aggregates, day-of-week reporting bias, and 7-day rolling trends |
| `m4_1d_analysis.png` | 4 | Histogram with central tendencies, boxplots, violin plots, and eCDF |
| `m4_2d_analysis.png` | 4 | Hexbin density plot, 2D Gaussian KDE contours, and Normal Q-Q plot |
| `m4_contingency_tables.png` | 4 | Cross-tabulation heatmap and Chi-Square independence analysis |
| `m4_nd_analysis.png` | 4 | Parallel coordinates, 3D scatter plot, and radar/spider profile chart |
| `m4_bar_dot_charts.png` | 4 | Grouped bar charts, horizontal bars, and Cleveland dot plot of CFR |
| `m5_clustering_results.png` | 5 | PCA scatter of Spectral, GMM, Agglomerative, and DBSCAN clusters |
| `m5_dendrogram.png` | 5 | Hierarchical clustering dendrogram with Cophenetic validation |
| `m5_model_selection.png` | 5 | BIC and AIC model selection curves across Gaussian mixture components |
| `m6_pca_svd.png` | 6 | Scree plot, cumulative variance curve, and PCA biplot / feature loadings |
| `m6_factor_analysis.png` | 6 | Factor loadings heatmap, communalities, and specificities |
| `m6_nonlinear_methods.png` | 6 | MDS projection, t-SNE (perplexities 5, 15, 30), and SOM hit-map |
| `m7_regression_evaluation.png` | 7 | Fitted regression lines, actual vs. predicted scatter, and error distributions |
| `m7_residual_diagnostics.png` | 7 | Residuals vs. fitted, normal Q-Q plot, autocorrelation, and Cook's distance |
| `m7_learning_cv_rl.png` | 7 | Learning curves, 5-fold CV comparisons, and RL Q-learning reward progress |

---

## 5. Environment & Dependencies

- Python 3.11+
- `pandas` $\ge$ 2.0.0
- `numpy` $\ge$ 1.24.0
- `scipy` $\ge$ 1.10.0
- `scikit-learn` $\ge$ 1.3.0
- `statsmodels` $\ge$ 0.14.0
- `matplotlib` $\ge$ 3.8.0
- `seaborn` $\ge$ 0.12.0
- `pymongo` $\ge$ 4.3.0
- `dash` $\ge$ 2.9.0
