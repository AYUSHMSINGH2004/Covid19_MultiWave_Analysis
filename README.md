# COVID-19 Multiwave Epidemiological Analysis & Machine Learning

<p align="center">
  <strong>An End-to-End Data Science, Statistical Analysis & Machine Learning Project</strong>
  <br />
  Exploring COVID-19 pandemic patterns, epidemiological trends, and multiwave behaviour through data-driven techniques.
  <br /><br />
  <a href="https://github.com/AYUSHMSINGH2004/Covid19_MultiWave_Analysis">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/Curriculum-Modules%202--7-6C63FF?style=for-the-badge" alt="Modules 2–7" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Implemented-2EA44F?style=for-the-badge" alt="Machine Learning" />
</p>

---

## Overview

The **COVID-19 Multiwave Epidemiological Analysis & Machine Learning Project** is a comprehensive data science implementation focused on analysing the progression, distribution, and multiwave characteristics of the COVID-19 pandemic.

The project brings together statistical computing, data preprocessing, exploratory data analysis, time-series modelling, unsupervised learning, dimensionality reduction, predictive modelling, and reinforcement learning into one unified framework.

It implements the complete practical curriculum from **Module 2 through Module 7**, with a modular architecture that enables individual module execution as well as automated end-to-end execution.

### Project Highlights

| Feature | Implementation |
|:---|:---|
| Curriculum Coverage | Modules 2–7 |
| Data Processing | Cleaning, transformation, binning and imputation |
| Statistical Analysis | Univariate, bivariate and multivariate analysis |
| Time Series Analysis | Decomposition, stationarity, ACF and PACF |
| Clustering | Spectral, GMM, MST, Agglomerative and DBSCAN |
| Dimensionality Reduction | PCA, SVD, Factor Analysis, MDS, t-SNE and SOM |
| Predictive Modelling | Linear, Multiple and ARDL Regression |
| Model Evaluation | Cross-validation, residual analysis and performance metrics |
| Reinforcement Learning | Q-Learning epidemic intervention simulation |
| Visualization | 23 automatically generated module-specific figures |
| Dashboard | Interactive multiwave analytics with MongoDB integration |

---

## Table of Contents

1. [Project Overview](#overview)
2. [Project Objectives](#project-objectives)
3. [Technology Stack](#technology-stack)
4. [Project Architecture](#project-architecture)
5. [Quick Start](#quick-start)
6. [Curriculum Implementation](#curriculum-implementation)
   - [Module 2 – Data Transformation](#module-2--data-transformation)
   - [Module 3 – Correlation and Time Series Analysis](#module-3--correlation-analysis-and-time-series-analysis)
   - [Module 4 – Data Summarization and Visualization](#module-4--data-summarization-and-visualization)
   - [Module 5 – Clustering Algorithms](#module-5--clustering-algorithms)
   - [Module 6 – Dimensionality Reduction](#module-6--dimensionality-reduction)
   - [Module 7 – Model Development and Evaluation](#module-7--model-development-and-evaluation)
7. [Execution Results](#execution-results)
8. [Generated Visualizations](#generated-visualizations)
9. [Interactive Dashboard](#interactive-dashboard)
10. [Environment and Dependencies](#environment-and-dependencies)
11. [Project Contributors](#project-contributors)
12. [Project Outcomes](#project-outcomes)
13. [Conclusion](#conclusion)

---

## Project Objectives

The primary objective is to apply data science and machine learning methodologies to real-world epidemiological data to understand pandemic behaviour and uncover meaningful patterns.

The project aims to:

- Perform comprehensive data cleaning and preprocessing.
- Investigate statistical relationships between confirmed cases and deaths.
- Analyse pandemic trends and identify multiwave patterns across countries.
- Apply clustering algorithms to identify groups with similar epidemiological characteristics.
- Reduce high-dimensional datasets while retaining important information.
- Develop regression models for epidemiological prediction.
- Evaluate model performance using statistical metrics and validation techniques.
- Demonstrate reinforcement learning through an epidemic intervention simulation.
- Present analytical findings through visualizations and an interactive dashboard.

---

## Technology Stack

The project is implemented using Python and its scientific computing ecosystem.

| Technology | Application |
|:---|:---|
| Python 3.11+ | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computation |
| SciPy | Statistical analysis and signal processing |
| Scikit-learn | Machine learning and clustering |
| Statsmodels | Regression and time-series modelling |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| PyMongo | MongoDB connectivity |
| Streamlit / Dash | Interactive dashboard components |
| MongoDB | Epidemiological data storage and aggregation |

---

## Project Architecture

The repository follows a modular structure that separates curriculum implementations, shared utilities, datasets, dashboard functionality, and generated visualizations.

```text
Covid19_MultiWave_Analysis/
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

### Directory Description

| Directory / File | Responsibility |
|:---|:---|
| `modules/` | Contains the six curriculum implementation modules. |
| `utils/` | Provides shared data-loading and feature-engineering utilities. |
| `data/` | Stores global confirmed cases and deaths datasets. |
| `dashboard/` | Contains interactive analytics and MongoDB integration scripts. |
| `plots/` | Stores automatically generated analytical figures. |
| `run_all_modules.py` | Master script for sequential execution of Modules 2–7. |
| `requirements.txt` | Contains the Python package dependencies. |

---

## Quick Start

The project provides a single master execution script to run all six curriculum modules sequentially.

### 1. Clone the Repository

```bash
git clone https://github.com/AYUSHMSINGH2004/Covid19_MultiWave_Analysis.git
```

### 2. Navigate to the Project Directory

```bash
cd Covid19_MultiWave_Analysis
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.\venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Execute the Complete Project

```bash
python run_all_modules.py
```

**Alternatively, on Windows, run directly using the virtual environment:**

```bash
.\venv\Scripts\python.exe run_all_modules.py
```

The master runner automatically executes Modules 2–7 and generates the corresponding analytical visualizations in the `plots/` directory.

---

# Curriculum Implementation

## Module 2 – Data Transformation

This module focuses on preparing raw epidemiological data for statistical analysis and machine learning.

### Key Implementations

**1. Data Deduplication**
- Exact duplicate detection using `df.duplicated()`.
- Logical duplicate identification using composite keys such as `Country/Region` and `Date`.
- Automated duplicate removal and data-cleaning demonstrations.

**2. Replacing Values**
- Handling zero-case values during early reporting periods.
- Statistical outlier detection using the \( \mu + 3\sigma \) threshold.
- Median-based outlier replacement.
- Transformation of daily case counts into severity categories:
  `Negligible`, `Low`, `Moderate`, and `Severe`.

**3. Discretization and Binning**
- Equal-width binning using `pd.cut()` with five intervals.
- Equal-frequency binning using `pd.qcut()` with four quartiles.
- Domain-specific categories: `Minimal`, `Controlled`, `Spreading`, `Surge`, and `Crisis`.

**4. Missing Data and Imputation**
- Mean imputation.
- Median imputation.
- Forward Fill (`ffill`).
- Backward Fill (`bfill`).
- Linear time-series interpolation.
- Maximum Likelihood Estimation (MLE) using the Expectation-Maximization (EM) approach.

Simplified iterative mean estimation:

\[
\mu^{(t+1)} =
\frac{1}{N}
\left(
\sum_{x_i \text{ observed}} x_i +
\sum_{x_j \text{ missing}} \mu^{(t)}
\right)
\]

**5. Imputation Evaluation**
- RMSE-based comparison of imputation methods.
- Performance comparison against known ground-truth values.

---

## Module 3 – Correlation Analysis and Time Series Analysis

This module examines statistical relationships between epidemiological variables and investigates the temporal evolution of the pandemic.

### Key Implementations

**1. Univariate Analysis**
- Skewness and kurtosis.
- Kernel Density Estimation (KDE).
- Histograms and boxplots.

**2. Bivariate Analysis**

Investigation of the relationship between daily confirmed cases and daily deaths.

| Correlation Method | Reported Value |
|:---|---:|
| Pearson Correlation | 0.697 |
| Spearman Rank Correlation | 0.771 |
| Kendall's Tau | 0.589 |

**3. Multivariate Analysis**
- Cross-country correlation matrix covering ten major nations.
- Pairwise joint distributions.
- Multivariate relationship analysis.

**4. Time Series Analysis**
- Additive and multiplicative time-series decomposition.
- Extraction of trend, seven-day seasonality, and residual components.
- Rolling 30-day mean and variance for stationarity analysis.
- Autocorrelation Function (ACF).
- Partial Autocorrelation Function (PACF).
- Lag-1 and lag-7 temporal dependency analysis.

**5. Time-Based Indexing and Resampling**
- Monthly grouping.
- Weekly resampling using `W`.
- Day-of-week reporting pattern analysis.
- Seven-day rolling average and smoothing.

---

## Module 4 – Data Summarization and Visualization

This module uses descriptive statistics and graphical methods to extract meaningful information from epidemiological datasets.

### Key Implementations

**1. Statistical Summary Measures**
- Count, mean, median, and mode.
- Variance and standard deviation.
- Range and interquartile range (IQR).
- Skewness and kurtosis.
- Coefficient of variation.
- Percentile analysis from \(P_1\) to \(P_{99}\).

**2. One-Dimensional Analysis**
- Histograms with mean and median markers.
- Comparative boxplots.
- Violin plots.
- Empirical Cumulative Distribution Functions (eCDF).

**3. Two-Dimensional Analysis**
- Hexagonal binning density plots.
- Two-dimensional Gaussian KDE contours.
- Quantile-Quantile (Q-Q) probability plots.

**4. Contingency Tables and Independence Testing**
- Case-severity versus death-severity contingency table.
- Pearson's Chi-Square Test of Independence.

Reported statistical results:

| Metric | Reported Value |
|:---|---:|
| Chi-Square Statistic | 1,437.13 |
| P-value | \(7.34 \times 10^{-304}\) |
| Degrees of Freedom | 9 |
| Cramér's V | 0.6474 |

**5. Multidimensional Analysis**
- Parallel coordinates charts.
- Three-dimensional scatter plots using cumulative cases, cumulative deaths, and Case Fatality Rate (CFR).
- Radar charts comparing multidimensional wave-severity profiles.

**6. Additional Visualizations**
- Cleveland dot plots for CFR comparisons.
- Grouped bar charts.
- Horizontal bar charts for cumulative morbidity.

---

## Module 5 – Clustering Algorithms

This module applies unsupervised learning to identify groups, patterns, and anomalies in epidemiological data.

### Key Implementations

**1. Spectral Clustering**
- RBF similarity graph construction.
- Graph Laplacian computation.
- Normalized-cut spectral embedding.

**2. Document Clustering**
- TF-IDF feature extraction from epidemic research abstracts.
- K-Means text clustering.
- Silhouette-based evaluation.

**3. Minimum Spanning Tree (MST) Clustering**
- Pairwise Euclidean distance computation.
- MST construction using Prim's or Kruskal's algorithm.
- Cluster formation through removal of the \(k-1\) longest tree edges.

**4. Gaussian Mixture Models (GMM)**
- Expectation-Maximization parameter estimation.
- Probabilistic clustering.
- Bayesian Information Criterion (BIC).
- Akaike Information Criterion (AIC).

**5. Hierarchical Agglomerative Clustering**
- Ward's minimum-variance linkage.
- Complete linkage.
- Dendrogram construction.
- Cophenetic Correlation Coefficient.

Reported cophenetic correlation:

\[
c = 0.814
\]

**6. Outlier Detection**
- Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
- Configuration: `min_samples = 3`, `eps = 1.5`.
- GMM-based anomaly detection through log-likelihood thresholding.

**7. Clustering Evaluation**

| Metric | Reported Value |
|:---|---:|
| Silhouette Score | 0.641 |
| Calinski-Harabasz Index | 141.2 |
| Davies-Bouldin Index | 0.548 |

---

## Module 6 – Dimensionality Reduction

This module implements linear and nonlinear dimensionality-reduction techniques to simplify high-dimensional data while preserving important information.

### Linear Dimensionality Reduction

**1. Principal Component Analysis (PCA)**
- Eigenvalue and eigenvector decomposition.
- Scree plot analysis.
- Cumulative explained variance.
- Feature loading analysis and PCA biplot.

Reported explained variance:

| Components | Explained Variance |
|:---|---:|
| PC1 | 58.3% |
| First 2 PCs | 74.8% |
| First 4 PCs | 94.7% |

**2. Singular Value Decomposition (SVD)**

\[
X = U\Sigma V^T
\]

- Economy SVD implementation.
- Rank-2, Rank-3, and Rank-5 approximations.
- Frobenius norm reconstruction error analysis.
- Reported reconstruction error at Rank-5: 12.1%.

**3. Factor Analysis**
- Maximum Likelihood Estimation of factor loadings.
- Uniqueness and specificity estimation.
- Communality calculation.
- Log-likelihood comparison across factor counts.

**4. Intrinsic Dimensionality Estimation**

| Estimation Technique | Estimated Dimension |
|:---|---:|
| PCA – 95% Variance Threshold | 5 |
| Kaiser Criterion | 3 |
| Levina-Bickel MLE | 4.22 |

### Nonlinear Dimensionality Reduction

**5. Multidimensional Scaling (MDS)**
- Pairwise distance preservation.
- Kruskal's Stress evaluation.

**6. t-SNE Manifold Learning**
- Nonlinear probability-distribution mapping.
- Experiments with perplexity values of 5, 15, and 30.
- Kullback-Leibler divergence minimization.

**7. Self-Organizing Maps (SOM)**
- \(6 \times 6\) Kohonen neural lattice.
- Two-dimensional Gaussian neighborhood adaptation.
- Country hit-map visualization.
- Quantization Error calculation.

Reported quantization error:

\[
QE = 0.4345
\]

---

## Module 7 – Model Development and Evaluation

This module develops predictive regression models and evaluates them through statistical metrics, validation methods, and residual diagnostics. It also demonstrates an epidemic intervention simulation using reinforcement learning.

### 1. Regression Models

**Model 1 – Simple Linear Regression**

- Predictor: Daily confirmed cases.
- Target: Daily deaths.

Reported performance:

| Metric | Value |
|:---|---:|
| R² | 0.234 |
| RMSE | 9,681 |

**Model 2 – Multiple Linear Regression**

- Uses a multi-feature predictor matrix.
- Predicts country-level mortality using epidemiological features.

**Model 3 – Autoregressive Distributed Lag (ARDL) Model**

- Incorporates 7-day, 14-day, and 21-day lagged case counts.
- Investigates delayed relationships between reported infections and mortality.

Reported performance:

| Metric | Value |
|:---|---:|
| R² | 0.735 |
| RMSE | 385.68 |

### 2. Comprehensive Evaluation Metrics

- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- Mean Absolute Error (MAE).
- Median Absolute Error.
- Mean Absolute Percentage Error (MAPE).
- Maximum Error.
- Coefficient of Determination (R²).
- Adjusted R².
- Explained Variance Score.
- Bias-Variance tradeoff.
- Regression metrics versus classification accuracy.

### 3. Residual Diagnostics

- Residuals versus fitted values.
- Heteroscedasticity analysis.
- Standardized residual Q-Q plots.
- Residual autocorrelation.
- Durbin-Watson test.
- Cook's Distance for influential observations.

### 4. Model Validation and Comparison

- Five-fold cross-validation.
- Mean and standard deviation of R², MAE, and RMSE.
- Learning curves.
- Training and cross-validation performance comparison.

Comparative model evaluation:

| Model | Technique |
|:---|:---|
| OLS | Ordinary Least Squares |
| Ridge | L2 Regularization |
| Lasso | L1 Regularization |
| Polynomial Regression | Degree 2 |

### 5. Reinforcement Learning Framework

The project also explores the fundamentals and practical applications of Reinforcement Learning.

**Theoretical Concepts**
- Supervised Learning versus Reinforcement Learning.
- Sequential decision-making.
- Feedback and reward mechanisms.
- Exploration versus exploitation.

**Q-Learning Epidemic Simulation**

A working Q-Learning simulation models pandemic intervention strategies as a Markov Decision Process (MDP).

Key components:
- Epidemic intervention policy modelling.
- Public-health containment versus economic lockdown penalties.
- Epsilon-greedy exploration.
- Q-table updates.
- Reward tracking and convergence analysis.

**Real-World Applications**
- AlphaGo.
- Robotics.
- Autonomous driving.
- Healthcare optimization.
- Dynamic epidemic management.
- Reinforcement Learning from Human Feedback (RLHF).

---

## Execution Results

The master execution script runs all six curriculum modules sequentially.

### Sample Execution Summary

```text
================================================================================
COVID-19 MULTI-WAVE ANALYSIS
COMPLETE MODULE EXECUTION (MODULES 2 -> 7)
================================================================================

>>> Running Module 2: Data Transformation                 [PASSED]
>>> Running Module 3: Correlation & Time Series Analysis  [PASSED]
>>> Running Module 4: Data Summarization & Visualization  [PASSED]
>>> Running Module 5: Clustering Algorithms               [PASSED]
>>> Running Module 6: Dimensionality Reduction            [PASSED]
>>> Running Module 7: Model Development & Evaluation      [PASSED]

================================================================================
EXECUTION SUMMARY
================================================================================

[PASSED] Module 2: Data Transformation
[PASSED] Module 3: Correlation & Time Series Analysis
[PASSED] Module 4: Data Summarization & Visualization
[PASSED] Module 5: Clustering Algorithms
[PASSED] Module 6: Dimensionality Reduction
[PASSED] Module 7: Model Development & Evaluation

Total time: 63.2 seconds
Generated plots: 23 files

================================================================================
ALL MODULES EXECUTED SUCCESSFULLY
================================================================================
```

**Execution Performance**

| Module | Execution Time |
|:---|---:|
| Module 2 – Data Transformation | 6.4 seconds |
| Module 3 – Correlation & Time Series | 17.9 seconds |
| Module 4 – Data Summarization & Visualization | 11.3 seconds |
| Module 5 – Clustering | 9.2 seconds |
| Module 6 – Dimensionality Reduction | 9.2 seconds |
| Module 7 – Model Development & Evaluation | 9.1 seconds |
| **Total** | **63.2 seconds** |

<sub>Note: These are representative execution results from the recorded run. Actual execution time may vary depending on hardware and environment.</sub>

---

## Generated Visualizations

The project automatically generates **23 module-specific visualization figures** inside the `plots/` directory.

| Visualization File | Module | Description |
|:---|:---:|:---|
| `m2_binning_distributions.png` | 2 | Equal-width, equal-frequency and domain-specific bin distributions |
| `m2_missing_data_imputation.png` | 2 | Comparison of traditional imputation and EM-based MLE |
| `m3_univariate_analysis.png` | 3 | Histogram, KDE and boxplot analysis |
| `m3_bivariate_analysis.png` | 3 | Scatter plot, OLS trendline and correlation analysis |
| `m3_multivariate_correlation.png` | 3 | Country-level correlation heatmaps |
| `m3_time_series_decomposition.png` | 3 | Trend, seasonality and residual decomposition |
| `m3_acf_pacf.png` | 3 | ACF and PACF correlograms |
| `m3_multi_country_timeseries.png` | 3 | Multi-country infection timelines |
| `m3_grouping_resampling.png` | 3 | Monthly aggregates and seven-day rolling trends |
| `m4_1d_analysis.png` | 4 | Histograms, boxplots, violin plots and eCDF |
| `m4_2d_analysis.png` | 4 | Hexbin density, KDE contours and Q-Q plots |
| `m4_contingency_tables.png` | 4 | Contingency heatmap and Chi-Square analysis |
| `m4_nd_analysis.png` | 4 | Parallel coordinates, 3D scatter and radar charts |
| `m4_bar_dot_charts.png` | 4 | Grouped bars, horizontal bars and Cleveland dot plots |
| `m5_clustering_results.png` | 5 | Comparison of clustering algorithm results |
| `m5_dendrogram.png` | 5 | Hierarchical clustering dendrogram |
| `m5_model_selection.png` | 5 | GMM model selection using BIC and AIC |
| `m6_pca_svd.png` | 6 | PCA scree plot, variance and feature loadings |
| `m6_factor_analysis.png` | 6 | Factor loadings, communalities and specificities |
| `m6_nonlinear_methods.png` | 6 | MDS, t-SNE and SOM visualizations |
| `m7_regression_evaluation.png` | 7 | Regression predictions and error distributions |
| `m7_residual_diagnostics.png` | 7 | Residual plots, Q-Q plots and Cook's Distance |
| `m7_learning_cv_rl.png` | 7 | Learning curves, cross-validation and Q-Learning |

**Additional Visualizations:** Five legacy multiwave detection plots are stored in `plots/legacy_waves/`, covering Brazil, India, Italy, the United States, and the United Kingdom.

---

## Interactive Dashboard

In addition to the curriculum modules, the repository includes a dashboard component for interactive exploration of COVID-19 epidemiological trends.

### Dashboard Components

| File | Functionality |
|:---|:---|
| `dashboard/app.py` | Interactive multiwave analytics dashboard |
| `dashboard/detect_waves.py` | Multiwave peak detection using SciPy signal analysis |
| `dashboard/ingest_data.py` | Bulk dataset ingestion into MongoDB |
| `dashboard/run_aggregation.py` | MongoDB aggregation pipelines for seven-day smoothing |

### Dashboard Capabilities

- Explore COVID-19 trends interactively.
- Analyse pandemic waves country-wise.
- Detect significant epidemic peaks.
- Visualize seven-day smoothed case trends.
- Store and aggregate epidemiological records using MongoDB.

The dashboard and database components can be configured and executed separately from the main curriculum runner.

---

## Environment and Dependencies

### System Requirements

| Requirement | Specification |
|:---|:---|
| Programming Language | Python 3.11+ |
| Package Manager | pip |
| Operating System | Windows / Linux / macOS |
| Database | MongoDB (for database-backed dashboard functionality) |

### Python Libraries

| Package | Minimum Version |
|:---|:---:|
| pandas | 2.0.0 |
| numpy | 1.24.0 |
| scipy | 1.10.0 |
| scikit-learn | 1.3.0 |
| statsmodels | 0.14.0 |
| matplotlib | 3.8.0 |
| seaborn | 0.12.0 |
| pymongo | 4.3.0 |
| dash | 2.9.0 |

All required packages are listed in `requirements.txt`.

---

## Project Contributors

This project was collaboratively developed by two team members, with an **equal distribution of responsibilities and an overall contribution split of 50% each**.

### Equal Contribution Matrix

| Project Responsibility | Ayush M Singh | Prem Prakash Puniya |
|:---|:---:|:---:|
| Module 2 – Data Transformation | Lead Implementation | Collaborative Review |
| Module 3 – Correlation & Time Series Analysis | Lead Implementation | Collaborative Review |
| Module 4 – Data Summarization & Visualization | Lead Implementation | Collaborative Review |
| Module 5 – Clustering Algorithms | Collaborative Review | Lead Implementation |
| Module 6 – Dimensionality Reduction | Collaborative Review | Lead Implementation |
| Module 7 – Model Development & Evaluation | Collaborative Review | Lead Implementation |
| Project Planning & Architecture | Shared | Shared |
| Dataset Understanding & Preparation | Shared | Shared |
| Module Integration | Shared | Shared |
| Testing & Validation | Shared | Shared |
| Debugging & Optimization | Shared | Shared |
| Documentation & Final Presentation | Shared | Shared |

### Contribution Breakdown

<table>
  <tr>
    <th align="center">Contributor</th>
    <th align="center">Registration Number</th>
    <th align="center">Primary Module Responsibilities</th>
    <th align="center">Contribution</th>
  </tr>
  <tr>
    <td><strong>Ayush M Singh</strong></td>
    <td align="center">23BDS0033</td>
    <td>Modules 2, 3 and 4</td>
    <td align="center"><strong>50%</strong></td>
  </tr>
  <tr>
    <td><strong>Prem Prakash Puniya</strong></td>
    <td align="center">23BDS0115</td>
    <td>Modules 5, 6 and 7</td>
    <td align="center"><strong>50%</strong></td>
  </tr>
  <tr>
    <td colspan="3"><strong>Total Team Contribution</strong></td>
    <td align="center"><strong>100%</strong></td>
  </tr>
</table>

**Shared Responsibilities:** Both members jointly contributed to project planning, dataset preparation, system integration, testing, validation, debugging, documentation, and final presentation.

The six curriculum modules are distributed equally, with three modules assigned as the primary implementation focus for each member. Both contributors participate in reviewing and integrating the complete project.

---

## Project Outcomes

The completed project demonstrates the practical application of statistical analysis, machine learning, and advanced analytical techniques to COVID-19 epidemiological data.

### Key Achievements

- Implemented six curriculum modules in a unified Python framework.
- Developed automated data transformation and imputation techniques.
- Performed correlation and multi-country time-series analysis.
- Applied multiple clustering techniques and validation metrics.
- Implemented linear and nonlinear dimensionality-reduction methods.
- Developed regression models and performed comprehensive model evaluation.
- Implemented a Q-Learning-based epidemic intervention simulation.
- Generated 23 module-specific analytical visualizations.
- Integrated an interactive dashboard with MongoDB-based data processing.

---

## Conclusion

The **COVID-19 Multiwave Epidemiological Analysis & Machine Learning Project** provides a comprehensive implementation of data science and machine learning concepts through real-world epidemiological data.

By integrating data transformation, statistical analysis, visualization, clustering, dimensionality reduction, predictive modelling, and reinforcement learning, the project demonstrates the complete analytical lifecycle from raw data preprocessing to model evaluation and interactive exploration.

Its modular and reproducible architecture provides a foundation for further experimentation, analytical extensions, and future epidemiological research.

---

## Repository Information

<p align="center">
  <a href="https://github.com/AYUSHMSINGH2004/Covid19_MultiWave_Analysis">
    <img src="https://img.shields.io/badge/Explore%20the%20Project-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Explore Project on GitHub" />
  </a>
</p>

**Repository:**  
https://github.com/AYUSHMSINGH2004/Covid19_MultiWave_Analysis.git

**Project Title:** COVID-19 Multiwave Epidemiological Analysis & Machine Learning

**Developed By:**

- **Ayush M Singh** — 23BDS0033
- **Prem Prakash Puniya** — 23BDS0115

<p align="center">
  <strong>Equal Contribution · 50% Each</strong>
  <br />
  <sub>Built through collaboration, experimentation, and applied data science.</sub>
</p>
