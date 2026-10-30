"""
Module 7: Model Development and Evaluation
============================================
Syllabus Topics:
  - Constructing linear regression model
  - Evaluation – computing accuracy – understanding accuracy
  - Understanding reinforcement learning:
      Difference between supervised and reinforcement learning
      Applications of reinforcement learning
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
from scipy import stats as sp_stats
from sklearn.model_selection import (train_test_split, cross_val_score,
                                       KFold, learning_curve)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                               r2_score, explained_variance_score,
                               median_absolute_error, max_error)

from utils.data_loader import (load_timeseries, load_country,
                          build_country_features, PLOTS_DIR)

print("=" * 80)
print("MODULE 7: MODEL DEVELOPMENT AND EVALUATION")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 7.1  CONSTRUCTING LINEAR REGRESSION MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("7.1 CONSTRUCTING LINEAR REGRESSION MODEL")
print("─" * 60)

# --- Dataset 1: Country-Level Regression (Total Cases → Total Deaths) ---
print("\n  7.1.1 Simple Linear Regression: Total Cases → Total Deaths")
features = build_country_features()
features = features[features["total_cases"] > 10000].copy()

X_country = features[["total_cases"]].values
y_country = features["total_deaths"].values
country_names = features["country"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_country, y_country, test_size=0.2, random_state=42)

lr_simple = LinearRegression()
lr_simple.fit(X_train, y_train)
y_pred_train = lr_simple.predict(X_train)
y_pred_test  = lr_simple.predict(X_test)

print(f"\n  Model: Deaths = {lr_simple.coef_[0]:.6f} × Cases + {lr_simple.intercept_:.2f}")
print(f"  Interpretation: For every 1,000 additional cases, ~{lr_simple.coef_[0]*1000:.1f} "
      f"additional deaths")

# --- Dataset 2: Multiple Linear Regression ---
print("\n  7.1.2 Multiple Linear Regression: Multi-Feature → Total Deaths")

feature_cols = ["total_cases", "peak_daily_cases", "mean_daily_cases",
                "std_daily_cases", "case_volatility"]
X_multi = features[feature_cols].values
y_multi = features["total_deaths"].values

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_ms = scaler.fit_transform(X_train_m)
X_test_ms  = scaler.transform(X_test_m)

lr_multi = LinearRegression()
lr_multi.fit(X_train_ms, y_train_m)
y_pred_multi = lr_multi.predict(X_test_ms)

print(f"\n  Coefficients (standardized):")
for feat, coef in zip(feature_cols, lr_multi.coef_):
    print(f"    {feat:>20s}: {coef:>12,.2f}")
print(f"    {'Intercept':>20s}: {lr_multi.intercept_:>12,.2f}")

# --- Dataset 3: Time-Series Regression for India ---
print("\n  7.1.3 Time-Series Regression: India daily_new_cases → daily_deaths (lagged)")

india = load_country("India")
india_reg = india[["daily_new_cases", "daily_deaths"]].copy()
# Use lagged cases as predictors (deaths lag behind cases)
for lag in [7, 14, 21]:
    india_reg[f"cases_lag_{lag}"] = india_reg["daily_new_cases"].shift(lag)
india_reg = india_reg.dropna()

X_india = india_reg[["daily_new_cases", "cases_lag_7", "cases_lag_14", "cases_lag_21"]].values
y_india = india_reg["daily_deaths"].values

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_india, y_india, test_size=0.2, random_state=42)

lr_india = LinearRegression()
lr_india.fit(X_train_i, y_train_i)
y_pred_india = lr_india.predict(X_test_i)

print(f"  Coefficients:")
for feat, coef in zip(["current_cases", "lag_7", "lag_14", "lag_21"], lr_india.coef_):
    print(f"    {feat:>15s}: {coef:.6f}")
print(f"    {'Intercept':>15s}: {lr_india.intercept_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.2  EVALUATION — COMPUTING ACCURACY — UNDERSTANDING ACCURACY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("7.2 MODEL EVALUATION & ACCURACY METRICS")
print("─" * 60)

def evaluate_model(name, y_true, y_pred, X_test=None, model=None):
    """Compute and print comprehensive regression evaluation metrics."""
    n = len(y_true)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    evs  = explained_variance_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    max_err = max_error(y_true, y_pred)

    # Adjusted R² (if we know the number of predictors)
    if X_test is not None:
        p = X_test.shape[1]
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2
    else:
        adj_r2 = r2

    print(f"\n  === {name} ===")
    print(f"    {'Metric':<30s} {'Value':>15s}")
    print(f"    {'─' * 46}")
    print(f"    {'Mean Squared Error (MSE)':<30s} {mse:>15,.2f}")
    print(f"    {'Root MSE (RMSE)':<30s} {rmse:>15,.2f}")
    print(f"    {'Mean Absolute Error (MAE)':<30s} {mae:>15,.2f}")
    print(f"    {'Median Absolute Error':<30s} {med_ae:>15,.2f}")
    print(f"    {'Mean Abs % Error (MAPE)':<30s} {mape:>14.2f}%")
    print(f"    {'Max Error':<30s} {max_err:>15,.2f}")
    print(f"    {'R² Score':<30s} {r2:>15.6f}")
    print(f"    {'Adjusted R²':<30s} {adj_r2:>15.6f}")
    print(f"    {'Explained Variance':<30s} {evs:>15.6f}")

    return {"name": name, "MSE": mse, "RMSE": rmse, "MAE": mae,
            "MAPE": mape, "R²": r2, "Adj R²": adj_r2}

# Evaluate all three models
m1 = evaluate_model("Simple LR (Cases → Deaths)",
                      y_test, y_pred_test, X_test, lr_simple)
m2 = evaluate_model("Multiple LR (Multi-Feature → Deaths)",
                      y_test_m, y_pred_multi, X_test_ms, lr_multi)
m3 = evaluate_model("Time-Series LR (India lagged)",
                      y_test_i, y_pred_india, X_test_i, lr_india)

# --- Cross-Validation ---
print("\n" + "─" * 60)
print("7.2.2 CROSS-VALIDATION ANALYSIS")
print("─" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, X_cv, y_cv in [
    ("Simple LR", X_country, y_country),
    ("Multiple LR", X_multi, y_multi),
    ("India Time-Series LR", X_india, y_india)
]:
    cv_r2   = cross_val_score(LinearRegression(), X_cv, y_cv, cv=kf, scoring="r2")
    cv_mae  = -cross_val_score(LinearRegression(), X_cv, y_cv, cv=kf,
                                scoring="neg_mean_absolute_error")
    cv_rmse = np.sqrt(-cross_val_score(LinearRegression(), X_cv, y_cv, cv=kf,
                                        scoring="neg_mean_squared_error"))

    print(f"\n  {name}:")
    print(f"    R²   folds: {[f'{v:.4f}' for v in cv_r2]}")
    print(f"    R²   mean ± std: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"    MAE  mean ± std: {cv_mae.mean():,.2f} ± {cv_mae.std():,.2f}")
    print(f"    RMSE mean ± std: {cv_rmse.mean():,.2f} ± {cv_rmse.std():,.2f}")

# --- Comparison: Linear vs Ridge vs Lasso vs Polynomial ---
print("\n" + "─" * 60)
print("7.2.3 MODEL COMPARISON (Linear vs Ridge vs Lasso vs Polynomial)")
print("─" * 60)

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge (α=1.0)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Lasso (α=1.0)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=1.0))
    ]),
    "Polynomial (deg=2)": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
}

comparison_results = []
for name, pipeline in models.items():
    cv_r2 = cross_val_score(pipeline, X_country, y_country, cv=kf, scoring="r2")
    cv_rmse = np.sqrt(-cross_val_score(pipeline, X_country, y_country, cv=kf,
                                        scoring="neg_mean_squared_error"))

    pipeline.fit(X_train, y_train)
    y_p = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_p)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_p))

    print(f"  {name:>25s}: CV R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}, "
          f"Test R² = {test_r2:.4f}, Test RMSE = {test_rmse:,.0f}")

    comparison_results.append({
        "Model": name,
        "CV R² (mean)": round(cv_r2.mean(), 4),
        "CV R² (std)": round(cv_r2.std(), 4),
        "Test R²": round(test_r2, 4),
        "Test RMSE": round(test_rmse, 2)
    })

comp_df = pd.DataFrame(comparison_results)
print(f"\n  Model Comparison Summary:")
print(comp_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7.3  UNDERSTANDING REINFORCEMENT LEARNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("7.3 UNDERSTANDING REINFORCEMENT LEARNING")
print("─" * 60)

rl_comparison = """
  ┌──────────────────────┬────────────────────────────────┬──────────────────────────────────┐
  │ Aspect               │ Supervised Learning            │ Reinforcement Learning           │
  ├──────────────────────┼────────────────────────────────┼──────────────────────────────────┤
  │ Training Signal      │ Labeled examples (input→output)│ Rewards/penalties (delayed)       │
  │ Feedback             │ Immediate, correct answer known│ Delayed, sparse, scalar reward    │
  │ Data Source          │ Static, pre-collected dataset  │ Agent interacts with environment  │
  │ Goal                 │ Minimize prediction error      │ Maximize cumulative reward        │
  │ Exploration          │ Not needed                     │ Explore vs exploit trade-off      │
  │ Sequential Decisions │ Independent predictions        │ Actions affect future states      │
  │ Examples             │ Image classification, NLP, LR  │ Game AI, robotics, trading        │
  │ COVID-19 Context     │ Predict deaths from cases      │ Optimize intervention policies    │
  └──────────────────────┴────────────────────────────────┴──────────────────────────────────┘
"""
print(rl_comparison)

# Simple Q-learning demonstration (Epidemic Intervention Simulation)
print("  7.3.1 Q-Learning Demo: Epidemic Intervention Simulation")
print("  " + "─" * 55)

class EpidemicEnvironment:
    """Simplified epidemic environment for RL demonstration.
    States: infection level (0-9), Actions: 0=No Intervention, 1=Lockdown
    Reward: -infection_level (penalize high infections), -3 for lockdown (economic cost)
    """
    def __init__(self):
        self.n_states = 10
        self.n_actions = 2
        self.state = 5  # Start at medium infection

    def reset(self):
        self.state = 5
        return self.state

    def step(self, action):
        if action == 0:  # No intervention
            # Infections tend to increase
            delta = np.random.choice([-1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.3])
        else:  # Lockdown
            # Infections tend to decrease
            delta = np.random.choice([-2, -1, 0, 1], p=[0.3, 0.4, 0.2, 0.1])

        self.state = int(np.clip(self.state + delta, 0, self.n_states - 1))

        # Reward: penalize infections + lockdown cost
        reward = -self.state + (-3 if action == 1 else 0)
        done = False
        return self.state, reward, done

env = EpidemicEnvironment()
Q = np.zeros((env.n_states, env.n_actions))

# Q-learning hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.95     # Discount factor
epsilon = 1.0    # Exploration rate
epsilon_decay = 0.995
n_episodes = 2000

episode_rewards = []
for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    for step in range(50):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = env.step(action)

        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',·)) - Q(s,a)]
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    epsilon *= epsilon_decay

print(f"\n    Training: {n_episodes} episodes")
print(f"    Final ε: {epsilon:.4f}")
print(f"    Avg Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
print(f"    Avg Reward (first 100 episodes): {np.mean(episode_rewards[:100]):.2f}")

print(f"\n    Learned Q-Table:")
print(f"    {'State':>8s}  {'No Action':>12s}  {'Lockdown':>12s}  {'Optimal':>10s}")
print(f"    {'─' * 46}")
for s in range(env.n_states):
    opt = "No Action" if Q[s, 0] > Q[s, 1] else "Lockdown"
    print(f"    {s:>8d}  {Q[s, 0]:>12.2f}  {Q[s, 1]:>12.2f}  {opt:>10s}")

print(f"\n    Policy: Agent learns to apply lockdowns when infections are high (≥ threshold)")
print(f"    This demonstrates how RL can optimize sequential decisions")
print(f"    (intervention timing) vs. supervised learning's static predictions.")

print(f"\n  7.3.2 Applications of Reinforcement Learning:")
applications = [
    ("Game Playing",        "AlphaGo, Atari, chess (DeepMind)"),
    ("Robotics",            "Robot locomotion, manipulation, navigation"),
    ("Autonomous Vehicles", "Self-driving decision making, lane changes"),
    ("Healthcare",          "Treatment regimens, drug dosing optimization"),
    ("Finance",             "Algorithmic trading, portfolio management"),
    ("NLP / LLMs",          "RLHF for ChatGPT/Gemini alignment"),
    ("Epidemic Control",    "Optimal intervention timing and intensity"),
    ("Resource Management", "Cloud auto-scaling, energy grid optimization"),
]
for area, desc in applications:
    print(f"    • {area:25s} → {desc}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Figure 1: Regression Results
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Module 7 — Linear Regression: Model Development & Evaluation",
             fontsize=14, fontweight="bold")

# 7a: Simple LR fit
axes[0, 0].scatter(X_test, y_test, s=30, alpha=0.6, c="#3b82f6",
                     edgecolors="black", linewidths=0.3, label="Test Data")
x_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
axes[0, 0].plot(x_range, lr_simple.predict(x_range), "r-", linewidth=2,
                 label=f"R² = {r2_score(y_test, y_pred_test):.4f}")
axes[0, 0].set_xlabel("Total Cases")
axes[0, 0].set_ylabel("Total Deaths")
axes[0, 0].set_title("Simple LR: Cases → Deaths")
axes[0, 0].legend()

# 7b: Actual vs Predicted
axes[0, 1].scatter(y_test, y_pred_test, s=30, alpha=0.6, c="#2563eb")
max_val = max(y_test.max(), y_pred_test.max())
axes[0, 1].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
axes[0, 1].set_xlabel("Actual Deaths")
axes[0, 1].set_ylabel("Predicted Deaths")
axes[0, 1].set_title("Actual vs Predicted (Simple LR)")
axes[0, 1].legend()

# 7c: Residual Plot
residuals = y_test - y_pred_test
axes[0, 2].scatter(y_pred_test, residuals, s=30, alpha=0.6, c="#16a34a")
axes[0, 2].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0, 2].set_xlabel("Predicted Deaths")
axes[0, 2].set_ylabel("Residuals (Actual - Predicted)")
axes[0, 2].set_title("Residual Analysis")

# 7d: Multiple LR — Actual vs Predicted
axes[1, 0].scatter(y_test_m, y_pred_multi, s=30, alpha=0.6, c="#f59e0b")
max_val2 = max(y_test_m.max(), y_pred_multi.max())
axes[1, 0].plot([0, max_val2], [0, max_val2], "r--", linewidth=2,
                 label=f"R² = {r2_score(y_test_m, y_pred_multi):.4f}")
axes[1, 0].set_xlabel("Actual Deaths")
axes[1, 0].set_ylabel("Predicted Deaths")
axes[1, 0].set_title("Multiple LR: Multi-Feature → Deaths")
axes[1, 0].legend()

# 7e: India Time-Series Actual vs Predicted (subset)
idx_sorted = np.argsort(y_test_i)[:100]
axes[1, 1].plot(range(len(idx_sorted)), y_test_i[idx_sorted], "b-",
                 linewidth=1.5, label="Actual", alpha=0.8)
axes[1, 1].plot(range(len(idx_sorted)), y_pred_india[idx_sorted], "r--",
                 linewidth=1.5, label="Predicted", alpha=0.8)
axes[1, 1].set_xlabel("Sample Index (sorted)")
axes[1, 1].set_ylabel("Daily Deaths")
axes[1, 1].set_title(f"India Lagged LR (R² = {r2_score(y_test_i, y_pred_india):.4f})")
axes[1, 1].legend()

# 7f: Model Comparison Bar Chart
comp_df_sorted = comp_df.sort_values("Test R²", ascending=True)
colors = ["#ef4444", "#f59e0b", "#3b82f6", "#22c55e"]
axes[1, 2].barh(comp_df_sorted["Model"], comp_df_sorted["Test R²"],
                  color=colors, edgecolor="black", linewidth=0.3)
axes[1, 2].set_xlabel("Test R² Score")
axes[1, 2].set_title("Model Comparison")
for i, (_, row) in enumerate(comp_df_sorted.iterrows()):
    axes[1, 2].text(row["Test R²"] + 0.01, i,
                      f"{row['Test R²']:.4f}", va="center", fontsize=9)

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m7_regression_evaluation.png"), dpi=200)
plt.close()
print(f"\n  ✓ Saved: plots/m7_regression_evaluation.png")

# Figure 2: Learning Curve + Cross-Validation + RL
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Module 7 — Cross-Validation, Learning Curve & RL Training",
             fontsize=14, fontweight="bold")

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X_country, y_country,
    cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="r2", random_state=42
)
axes[0].plot(train_sizes, train_scores.mean(axis=1), "o-", color="#2563eb",
              linewidth=2, label="Training R²")
axes[0].fill_between(train_sizes,
                       train_scores.mean(axis=1) - train_scores.std(axis=1),
                       train_scores.mean(axis=1) + train_scores.std(axis=1),
                       alpha=0.15, color="#2563eb")
axes[0].plot(train_sizes, test_scores.mean(axis=1), "o-", color="#dc2626",
              linewidth=2, label="Validation R²")
axes[0].fill_between(train_sizes,
                       test_scores.mean(axis=1) - test_scores.std(axis=1),
                       test_scores.mean(axis=1) + test_scores.std(axis=1),
                       alpha=0.15, color="#dc2626")
axes[0].set_xlabel("Training Set Size")
axes[0].set_ylabel("R² Score")
axes[0].set_title("Learning Curve (Simple LR)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cross-validation R² distribution
cv_all = cross_val_score(LinearRegression(), X_country, y_country, cv=10, scoring="r2")
axes[1].bar(range(1, 11), cv_all, color="#60a5fa", edgecolor="black", linewidth=0.3)
axes[1].axhline(cv_all.mean(), color="#dc2626", linestyle="--",
                 label=f"Mean = {cv_all.mean():.4f}")
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("R² Score")
axes[1].set_title("10-Fold Cross-Validation")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# RL Training Curve
window = 50
smoothed_rewards = pd.Series(episode_rewards).rolling(window).mean()
axes[2].plot(smoothed_rewards, color="#16a34a", linewidth=1.5,
              label=f"Moving Avg (w={window})")
axes[2].plot(episode_rewards, alpha=0.15, color="#16a34a", linewidth=0.5)
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Total Reward")
axes[2].set_title("Q-Learning: Epidemic Intervention Training")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m7_learning_cv_rl.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m7_learning_cv_rl.png")

# Figure 3: Residual Diagnostics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Module 7 — Residual Diagnostics (Simple LR)",
             fontsize=14, fontweight="bold")

# Residual Distribution
axes[0].hist(residuals, bins=30, density=True, color="#93c5fd",
              edgecolor="white", alpha=0.7)
mu, sigma = sp_stats.norm.fit(residuals)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[0].plot(x_norm, sp_stats.norm.pdf(x_norm, mu, sigma), "r-", linewidth=2)
axes[0].set_title("Residual Distribution")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Density")

# QQ plot
sp_stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot of Residuals")
axes[1].get_lines()[0].set_color("#2563eb")
axes[1].get_lines()[1].set_color("#dc2626")

# Residual vs Fitted (homoscedasticity check)
std_resid = residuals / residuals.std()
axes[2].scatter(y_pred_test, std_resid, alpha=0.5, s=30, c="#8b5cf6")
axes[2].axhline(0, color="red", linestyle="--")
axes[2].axhline(2, color="gray", linestyle=":", alpha=0.5)
axes[2].axhline(-2, color="gray", linestyle=":", alpha=0.5)
axes[2].set_xlabel("Predicted Values")
axes[2].set_ylabel("Standardized Residuals")
axes[2].set_title("Homoscedasticity Check")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "m7_residual_diagnostics.png"), dpi=200)
plt.close()
print(f"  ✓ Saved: plots/m7_residual_diagnostics.png")

# ── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("FINAL MODEL EVALUATION SUMMARY")
print("─" * 60)
final_summary = pd.DataFrame([m1, m2, m3])
print(final_summary[["name", "R²", "Adj R²", "RMSE", "MAE", "MAPE"]].to_string(index=False))

print("\n" + "=" * 80)
print("MODULE 7 COMPLETE")
print("=" * 80)
