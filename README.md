# COVID-19 Multiwave Epidemiological Analysis & Machine Learning Project

## Complete Curriculum Implementation: Modules 2–7

This repository contains an end-to-end data science and machine learning pipeline for multiwave COVID-19 pandemic analysis. It implements the course curriculum from **Module 2 to Module 7** within a unified, modular, and reproducible framework.

The project covers data transformation, statistical analysis, time-series analysis, data summarization, visualization, clustering, dimensionality reduction, model development, and evaluation. It also includes an interactive dashboard for exploring pandemic trends across multiple countries.

---

## 1. Quick Start: Single-Stretch Execution

Run the complete curriculum from Module 2 to Module 7 using the master execution script.

**Using the virtual environment:**

```bash
.\venv\Scripts\python.exe run_all_modules.py
```

**Or, after activating the virtual environment:**

```bash
python run_all_modules.py
```

### Execution Output & Verification

The `run_all_modules.py` script sequentially executes all six modules, displays analytical tables and evaluation metrics in the terminal, and generates **23 visualization files** in the `plots/` directory.

```text
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
  [PASSED] Module 2: Data Transformation                 (6.4s)
  [PASSED] Module 3: Correlation & Time Series Analysis  (17.9s)
  [PASSED] Module 4: Data Summarization & Visualization  (11.3s)
  [PASSED] Module 5: Clustering Algorithms               (9.2s)
  [PASSED] Module 6: Dimensionality Reduction            (9.2s)
  [PASSED] Module 7: Model Development & Evaluation      (9.1s)

  Total time: 63.2s
  Generated plots: 23 files in plots/
================================================================================
```

*Note: The execution output above represents the recorded example run. Actual execution time may vary depending on the system and environment.*

---

## 2. Project Architecture & File Organization

```text
COVID-19 Multiwave Analysis/
│
├── run_all_modules.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── module2_data_transformation.py
│   ├── module3_correlation_timeseries.py
│   ├── module4_summarization_visualization.py
│   ├── module5_clustering.py
│   ├── module6_dimensionality_reduction.py
│   └── module7_model_evaluation.py
│
├── utils/
│   ├── __init__.py
│   └── data_loader.py
│
├── data/
│   ├── time_series_covid19_confirmed_global.csv
│   └── time_series_covid19_deaths_global.csv
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   ├── detect_waves.py
│   ├── ingest_data.py
│   └── run_aggregation.py
│
└── plots/
    ├── legacy_waves/
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

### Key Components

| Component | Description |
|:---|:---|
| `run_all_modules.py` | Master execution script for running Modules 2–7 sequentially. |
| `modules/` | Contains the six curriculum implementation modules. |
| `utils/` | Shared data-loading and feature-engineering utilities. |
| `data/` | Contains the global confirmed cases and deaths datasets. |
| `dashboard/` | Contains the interactive dashboard and MongoDB integration scripts. |
| `plots/` | Stores generated visualizations and legacy multiwave plots. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |

---

## 3. Detailed Curriculum & Syllabus Breakdown

### Module 2: Data Transformation

This module focuses on cleaning, transforming, discretizing, and imputing epidemiological data.

**Data Deduplication**
- Exact duplicate detection using `df.duplicated()`.
- Logical duplicate detection using composite keys such as `Country/Region` and `Date`.
- Demonstration of duplicate insertion and automated cleanup.

**Replacing Values**
- Handling zero-case values during early reporting periods.
- Detecting statistical outliers using the threshold \(x > \mu + 3\sigma\) and replacing them with the median.
- Mapping daily case counts into epidemiological severity categories: `Negligible`, `Low`, `Moderate`, and `Severe`.

**Discretization & Binning**
- Equal-width binning using `pd.cut()` with five uniform intervals.
- Equal-frequency or quantile binning using `pd.qcut()` with four quartiles: Q1, Q2, Q3, and Q4.
- Domain-specific binning into `Minimal`, `Controlled`, `Spreading`, `Surge`, and `Crisis`.

**Missing Data & Imputation**
- Mean imputation.
- Median imputation.
- Forward fill (`ffill`).
- Backward fill (`bfill`).
- Linear time-series interpolation.
- Maximum Likelihood Estimation (MLE) using an Expectation-Maximization (EM) approach under a normal-distribution assumption.

The simplified iterative mean update is:

\[
\mu^{(t+1)} =
\frac{1}{N}
\left(
\sum_{x_i\text{ observed}} x_i +
\sum_{x_j\text{ missing}} \mu^{(t)}
\right)
\]

- Quantitative comparison of imputation methods using Root Mean Squared Error (RMSE) against known values.

---

### Module 3: Correlation Analysis and Time Series Analysis

This module investigates statistical relationships between COVID-19 cases and deaths, along with temporal patterns across countries.

**Univariate Analysis**
- Skewness and kurtosis.
- Kernel Density Estimation (KDE).
- Distribution analysis using histograms and boxplots.

**Bivariate Analysis**
- Relationship between daily confirmed cases and daily deaths.
- Pearson correlation: \(r = 0.697\).
- Spearman rank correlation: \(\rho = 0.771\).
- Kendall's Tau: \(\tau = 0.589\).

**Multivariate Analysis**
- Cross-country correlation matrix covering ten major nations.
- Pairwise joint distributions and multivariate relationship analysis.

**Time Series Analysis (TSA)**

- **Time-series decomposition:** Additive and multiplicative decomposition into observed, trend, seven-day seasonal, and residual components.
- **Stationarity analysis:** Rolling 30-day mean and variance to examine changes in pandemic trends.
- **Autocorrelation:** ACF and PACF plots to investigate serial dependencies, including lag-1 and lag-7 patterns.
- **Time-based indexing:** Date-based indexing and chronological data analysis.
- **Grouping and resampling:** Monthly grouping, weekly (`W`) resampling, day-of-week reporting patterns, and seven-day rolling averages.

---

### Module 4: Data Summarization and Visualization

This module applies descriptive statistics and visualization techniques to understand the distribution and relationships within the COVID-19 dataset.

**Statistical Summary Measures**
- Count, mean, median, and mode.
- Variance and standard deviation.
- Range and interquartile range (IQR).
- Skewness and kurtosis.
- Coefficient of variation.
- Percentile distribution from \(P_1\) to \(P_{99}\).

**1-D Statistical Analysis**
- Histograms with mean and median markers.
- Comparative boxplots.
- Violin plots.
- Empirical Cumulative Distribution Functions (eCDF).

**2-D Statistical Analysis**
- Hexagonal binning density plots.
- Two-dimensional bivariate Gaussian KDE contours.
- Quantile-Quantile (Q-Q) normal probability plots.

**Contingency Tables & Independence Testing**
- Two-way contingency tables comparing case severity and death severity.
- Pearson's Chi-Square Test of Independence:

\[
\chi^2 = 1437.13,\quad
p = 7.34 \times 10^{-304},\quad
DoF = 9
\]

- Effect-size measurement using Cramér's V:

\[
V = 0.6474
\]

**n-D Statistical Analysis**
- Parallel coordinates chart displaying normalized multidimensional country profiles.
- Three-dimensional scatter plot using cumulative cases, cumulative deaths, and Case Fatality Rate (CFR).
- Radar or spider chart comparing multidimensional wave-severity profiles across global hotspots.

**Additional Visualizations**
- Cleveland dot plots for Case Fatality Rates.
- Grouped bar charts.
- Horizontal bar charts for cumulative morbidity comparisons.

---

### Module 5: Clustering Algorithms

This module explores unsupervised learning techniques for identifying patterns and grouping countries based on epidemiological characteristics.

**Spectral Clustering**
- RBF similarity graph construction.
- Graph Laplacian computation.
- Normalized-cut spectral embedding.

**Document Clustering**
- TF-IDF feature extraction from epidemic research abstracts.
- K-Means clustering for text documents.
- Silhouette-based clustering evaluation.

**Minimum Spanning Tree (MST) Clustering**
- Pairwise Euclidean distance graph construction.
- MST generation using Prim's or Kruskal's algorithm.
- Cluster formation by removing the \(k-1\) longest edges from the spanning tree.

**Model-Based Clustering**
- Gaussian Mixture Models (GMM).
- Expectation-Maximization (EM) parameter estimation.
- Model-order selection using Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC).

**Hierarchical Agglomerative Clustering**
- Ward's minimum-variance linkage.
- Complete linkage.
- Dendrogram construction.
- Cophenetic Correlation Coefficient validation:

\[
c = 0.814
\]

**Outlier Detection**
- Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
- DBSCAN configuration: `min_samples = 3`, `eps = 1.5`.
- GMM-based anomaly detection using low log-likelihood density thresholds.

**Clustering Validation Metrics**
- Silhouette Score: \(S = 0.641\).
- Calinski-Harabasz Index: \(CH = 141.2\).
- Davies-Bouldin Index: \(DB = 0.548\).

---

### Module 6: Dimensionality Reduction

This module applies linear and nonlinear techniques to reduce feature dimensionality while retaining important information.

**Linear Methods**

**Principal Component Analysis (PCA)**
- Eigenvalue and eigenvector decomposition.
- Scree plot analysis.
- Cumulative explained variance:
  - PC1: 58.3%.
  - First two principal components: 74.8%.
  - First four principal components: 94.7%.
- Feature loading and PCA biplot visualization.

**Singular Value Decomposition (SVD)**
- Economy SVD:

\[
X = U\Sigma V^T
\]

- Low-rank matrix approximation using Rank-2, Rank-3, and Rank-5 models.
- Frobenius norm reconstruction error analysis.
- Reported reconstruction error at Rank-5: 12.1%.

**Factor Analysis**
- Maximum Likelihood Estimation of latent factor loadings.
- Uniqueness and specificity estimation.
- Communality calculation.
- Model log-likelihood comparison across different factor counts.

**Intrinsic Dimensionality Estimation**
- PCA 95% explained-variance threshold: \(d = 5\).
- Kaiser Criterion (eigenvalues greater than 1): \(d = 3\).
- Levina-Bickel Maximum Likelihood Estimation: \(d = 4.22\).

**Nonlinear Methods**

**Multidimensional Scaling (MDS)**
- Pairwise distance preservation.
- Kruskal's Stress formula for evaluating the quality of lower-dimensional representations.

**t-SNE Manifold Learning**
- Nonlinear probability-distribution mapping.
- Experiments with perplexity values of 5, 15, and 30.
- Kullback-Leibler (KL) divergence minimization.

**Self-Organizing Maps (SOM)**
- A \(6 \times 6\) Kohonen artificial neural lattice.
- Two-dimensional Gaussian neighborhood adaptation.
- Country hit-map and activation-frequency visualization.
- Quantization Error:

\[
QE = 0.4345
\]

---

### Module 7: Model Development and Evaluation

This module develops regression and time-series prediction models and evaluates their performance using statistical metrics, residual diagnostics, and cross-validation. It also demonstrates a Q-Learning-based epidemic intervention simulation.

**Linear Regression Models**

**Model 1 — Simple Linear Regression**
- Predictor: Daily confirmed cases.
- Target: Daily deaths.
- Reported performance:

\[
R^2 = 0.234,\quad RMSE = 9681
\]

**Model 2 — Multiple Linear Regression**
- Uses a full multi-feature predictor matrix.
- Predicts country-level mortality using multiple epidemiological features.

**Model 3 — Autoregressive Distributed Lag (ARDL) Time-Series Model**
- Incorporates seven-day, 14-day, and 21-day lagged case counts.
- Models delayed relationships between reported infections and mortality.
- Reported performance:

\[
R^2 = 0.735,\quad RMSE = 385.68
\]

**Comprehensive Evaluation Metrics**

- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- Mean Absolute Error (MAE).
- Median Absolute Error.
- Mean Absolute Percentage Error (MAPE).
- Maximum Error.
- Coefficient of Determination (\(R^2\)).
- Adjusted \(R^2\).
- Explained Variance Score.
- Regression evaluation versus classification accuracy.
- Bias-Variance tradeoff.

**Residual Diagnostics & Assumption Testing**
- Residuals versus fitted values for heteroscedasticity analysis.
- Normal Q-Q plot of standardized residuals.
- Residual autocorrelation analysis.
- Durbin-Watson test.
- Cook's Distance for identifying influential observations and high-leverage outliers.

**Model Validation & Comparisons**
- Five-fold cross-validation.
- Mean and standard deviation of \(R^2\), MAE, and RMSE.
- Learning curves comparing training and cross-validation performance across sample sizes.
- Comparative evaluation of:
  - Ordinary Least Squares (OLS).
  - Ridge Regression (\(L_2\) regularization).
  - Lasso Regression (\(L_1\) regularization).
  - Polynomial Regression (degree 2).

**Reinforcement Learning (RL) Framework**

- Theoretical comparison between supervised learning and reinforcement learning.
- Feedback mechanisms and sequential decision-making.
- Exploration versus exploitation.

**Q-Learning Epidemic Simulation**
- Models pandemic intervention strategies as a Markov Decision Process (MDP).
- Explores policies balancing public-health containment against economic lockdown penalties.
- Uses epsilon-greedy exploration.
- Tracks Q-table updates and convergence.

**Real-World Reinforcement Learning Applications**
- AlphaGo.
- Robotics.
- Autonomous driving.
- Healthcare and treatment optimization.
- Dynamic epidemic management.
- Reinforcement Learning from Human Feedback (RLHF) for Large Language Models (LLMs).

---

## 4. Complete List of Generated Visualizations

The 23 module-specific figures are automatically saved in the `plots/` directory.

| File Name | Module | Visualization Content |
|:---|:---:|:---|
| `m2_binning_distributions.png` | 2 | Equal-width, equal-frequency, and domain-specific bin distributions. |
| `m2_missing_data_imputation.png` | 2 | Traditional imputation methods versus EM-based MLE comparison. |
| `m3_univariate_analysis.png` | 3 | Histogram, KDE, and boxplot distribution analysis. |
| `m3_bivariate_analysis.png` | 3 | Scatter plot with OLS trendline and Pearson, Spearman, and Kendall correlations. |
| `m3_multivariate_correlation.png` | 3 | Cross-country correlation heatmaps and pairwise joint distributions. |
| `m3_time_series_decomposition.png` | 3 | Observed, trend, seven-day seasonality, and residual components. |
| `m3_acf_pacf.png` | 3 | Autocorrelation and Partial Autocorrelation correlograms. |
| `m3_multi_country_timeseries.png` | 3 | Multi-country chronological COVID-19 infection timelines. |
| `m3_grouping_resampling.png` | 3 | Monthly aggregates, day-of-week reporting patterns, and seven-day rolling trends. |
| `m4_1d_analysis.png` | 4 | Histograms, central tendencies, boxplots, violin plots, and eCDF. |
| `m4_2d_analysis.png` | 4 | Hexbin density plot, two-dimensional KDE contours, and normal Q-Q plot. |
| `m4_contingency_tables.png` | 4 | Contingency-table heatmap and Chi-Square independence analysis. |
| `m4_nd_analysis.png` | 4 | Parallel coordinates, 3D scatter plot, and radar/spider chart. |
| `m4_bar_dot_charts.png` | 4 | Grouped bar charts, horizontal bars, and Cleveland dot plot of CFR. |
| `m5_clustering_results.png` | 5 | PCA visualization of Spectral, GMM, Agglomerative, and DBSCAN clusters. |
| `m5_dendrogram.png` | 5 | Hierarchical clustering dendrogram with Cophenetic validation. |
| `m5_model_selection.png` | 5 | BIC and AIC model-selection curves across Gaussian mixture components. |
| `m6_pca_svd.png` | 6 | Scree plot, cumulative explained variance, and PCA feature-loading biplot. |
| `m6_factor_analysis.png` | 6 | Factor-loadings heatmap, communalities, and specificities. |
| `m6_nonlinear_methods.png` | 6 | MDS projection, t-SNE projections, and SOM hit-map. |
| `m7_regression_evaluation.png` | 7 | Regression lines, actual-versus-predicted scatter plots, and error distributions. |
| `m7_residual_diagnostics.png` | 7 | Residuals versus fitted values, normal Q-Q plot, autocorrelation, and Cook's Distance. |
| `m7_learning_cv_rl.png` | 7 | Learning curves, five-fold cross-validation comparisons, and Q-Learning reward progress. |

**Additional legacy visualizations:** The `plots/legacy_waves/` folder contains five country-specific multiwave detection plots for Brazil, India, Italy, the United States, and the United Kingdom.

---

## 5. Interactive Dashboard & Database Integration

The project also includes a dashboard component for interactive exploration of multiwave COVID-19 trends.

| File | Purpose |
|:---|:---|
| `dashboard/app.py` | Streamlit-based interactive multiwave analytics dashboard. |
| `dashboard/detect_waves.py` | Detects pandemic waves using SciPy signal-processing techniques. |
| `dashboard/ingest_data.py` | Performs bulk dataset ingestion into MongoDB. |
| `dashboard/run_aggregation.py` | Executes MongoDB aggregation pipelines for seven-day smoothing. |

The dashboard component supports:
- Interactive exploration of COVID-19 trends.
- Country-wise pandemic wave analysis.
- Peak detection and multiwave visualization.
- Seven-day smoothing of reported case trends.
- MongoDB-based storage and aggregation of epidemiological records.

---

## 6. Environment & Dependencies

**Prerequisites**
- Python 3.11 or higher.
- pip package manager.
- MongoDB, when using the database-backed dashboard functionality.

**Python Dependencies**

| Library | Minimum Version | Purpose |
|:---|:---:|:---|
| Python | 3.11+ | Programming environment. |
| pandas | 2.0.0 | Data manipulation and analysis. |
| numpy | 1.24.0 | Numerical computation. |
| scipy | 1.10.0 | Statistical analysis and signal processing. |
| scikit-learn | 1.3.0 | Machine learning, clustering, and dimensionality reduction. |
| statsmodels | 0.14.0 | Statistical modelling and time-series analysis. |
| matplotlib | 3.8.0 | Data visualization. |
| seaborn | 0.12.0 | Statistical visualization. |
| pymongo | 4.3.0 | MongoDB integration. |
| dash | 2.9.0 | Dashboard application support. |

### Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd "COVID-19 Multiwave Analysis"
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
.\venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the complete analysis:

```bash
python run_all_modules.py
```

*Note: Replace `<YOUR_REPOSITORY_URL>` with the actual Git repository URL. MongoDB configuration may also be required before running database-backed dashboard features.*

---

## 7. Project Contributors & Responsibilities

The project was developed collaboratively by two contributors, with responsibilities divided equally across the six curriculum modules. Both contributors also share responsibility for project planning, integration, testing, and documentation.

| Contributor | Registration Number | Primary Responsibilities |
|:---|:---:|:---|
| **Ayush M Singh** | **23BDS0033** | Module 2 – Data Transformation; Module 3 – Correlation & Time Series Analysis; Module 4 – Data Summarization & Visualization; shared project integration, testing, and documentation. |
| **Prem Prakash Puniya** | **23BDS0115** | Module 5 – Clustering Algorithms; Module 6 – Dimensionality Reduction; Module 7 – Model Development & Evaluation; shared project integration, testing, and documentation. |

### Shared Responsibilities

Both contributors collaboratively worked on:

- Project planning and overall architecture.
- Dataset understanding and selection.
- Integration of individual modules into the unified pipeline.
- Validation of analytical results and visualizations.
- Testing and debugging.
- Final project documentation and presentation.

**Contribution Distribution:** Each contributor is responsible for three of the six curriculum modules, with shared ownership of integration, testing, and documentation.

---

## 8. Project Outcomes

The project demonstrates an end-to-end application of data science and machine learning techniques to real-world epidemiological data.

Key outcomes include:

- A unified, modular implementation of Modules 2–7.
- Automated data cleaning, transformation, and missing-value imputation.
- Statistical analysis of COVID-19 cases, deaths, and cross-country relationships.
- Time-series decomposition and multiwave trend analysis.
- Descriptive statistical summaries and multidimensional visualizations.
- Implementation and evaluation of multiple clustering algorithms.
- Linear and nonlinear dimensionality-reduction techniques.
- Regression modelling, residual diagnostics, and cross-validation.
- A working Q-Learning simulation for epidemic intervention strategies.
- An interactive dashboard supported by MongoDB integration.
- 23 automatically generated module-specific visualization figures.

---

## 9. Conclusion

The **COVID-19 Multiwave Epidemiological Analysis & Machine Learning Project** brings together statistical analysis, data science, machine learning, time-series modelling, and reinforcement learning in a single reproducible framework.

By implementing the curriculum from **Module 2 through Module 7**, the project demonstrates how data preprocessing, statistical methods, unsupervised learning, dimensionality reduction, predictive modelling, and interactive visualization can be applied to understand the progression and behaviour of a global pandemic.

The modular architecture also makes the implementation easier to execute, maintain, validate, and extend for future epidemiological research and analytical applications.

---

**Developed by:**  
**Ayush M Singh (23BDS0033)**  
**Prem Prakash Puniya (23BDS0115)**

**Project:** COVID-19 Multiwave Epidemiological Analysis & Machine Learning Project
