"""
=============================================================================
 Telco Customer Churn — Step 9: ML Modelling & SHAP Analysis
=============================================================================

This step builds on the correctly preprocessed data from Steps 1-8.

Pipeline:
  1. Train 7 baseline models with proper CV (no leakage)
  2. Tune top 3 models with Optuna
  3. Build ensemble models
  4. Optimize decision threshold (F2 — recall-weighted for churn)
  5. Final evaluation on the LOCKED test set
  6. SHAP explainability analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier,
    StackingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_curve, auc, average_precision_score, fbeta_score, roc_auc_score,
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    recall_score, precision_score, f1_score, accuracy_score,
)
import xgboost as xgb
import optuna
import shap
import joblib
import os
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 11

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(ROOT_DIR, 'data', 'processed')
DATA_SPLITS = os.path.join(ROOT_DIR, 'data', 'splits')
ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'artifacts')
REPORTS_MODEL = os.path.join(ROOT_DIR, 'reports', 'modelling')
REPORTS_SHAP = os.path.join(ROOT_DIR, 'reports', 'shap')
os.makedirs(REPORTS_MODEL, exist_ok=True)
os.makedirs(REPORTS_SHAP, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================================
# LOAD PREPROCESSED DATA
# ============================================================================

print("=" * 70)
print(" STEP 9: ML MODELLING & SHAP ANALYSIS")
print("=" * 70)

X_train = pd.read_csv(os.path.join(DATA_SPLITS, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(DATA_SPLITS, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_SPLITS, 'y_train.csv')).squeeze()
y_test = pd.read_csv(os.path.join(DATA_SPLITS, 'y_test.csv')).squeeze()

scale_pw = (y_train == 0).sum() / (y_train == 1).sum()

print(f"\n  Training set: {X_train.shape[0]} rows x {X_train.shape[1]} features")
print(f"  Test set:     {X_test.shape[0]} rows x {X_test.shape[1]} features")
print(f"  Churn rate — train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")
print(f"  Class weight ratio: {scale_pw:.2f}:1\n")


# ============================================================================
# 9.1 BASELINE MODELS — 7 Algorithms with 5-Fold Stratified CV
# ============================================================================
#
# All models are evaluated using cross_val_predict on the TRAINING set only.
# The test set is untouched until final evaluation.
#
# Data is already scaled (Step 6), so all algorithms receive the same input.
# Models that need class balancing get class_weight='balanced' or equivalent.
# ============================================================================

print("=" * 70)
print(" 9.1 — BASELINE MODELS (5-Fold Stratified CV on Training Set)")
print("=" * 70)


def evaluate_model(name, y_true, y_proba):
    pr_auc = average_precision_score(y_true, y_proba)
    roc = roc_auc_score(y_true, y_proba)
    best_f2, best_t = 0, 0.5
    for t in np.arange(0.20, 0.80, 0.01):
        f2 = fbeta_score(y_true, (y_proba >= t).astype(int), beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t
    return {"model": name, "PR-AUC": pr_auc, "ROC-AUC": roc, "Best_F2": best_f2, "F2_threshold": best_t}


models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42, max_depth=6
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, random_state=42, max_depth=4
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200, random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, scale_pos_weight=scale_pw,
        eval_metric="aucpr", random_state=42, use_label_encoder=False, max_depth=4,
    ),
}

baseline_results = []
model_probas = {}

print(f"\n  {'Model':30s}  {'PR-AUC':>8s}  {'ROC-AUC':>8s}  {'Best F2':>8s}")
print(f"  {'─' * 62}")

for name, model in models.items():
    y_proba = cross_val_predict(model, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    res = evaluate_model(name, y_train, y_proba)
    baseline_results.append(res)
    model_probas[name] = y_proba
    print(f"  {name:30s}  {res['PR-AUC']:.4f}    {res['ROC-AUC']:.4f}    {res['Best_F2']:.4f}")

baseline_df = pd.DataFrame(baseline_results).sort_values("PR-AUC", ascending=False)
print(f"\n  Top performer: {baseline_df.iloc[0]['model']} (PR-AUC = {baseline_df.iloc[0]['PR-AUC']:.4f})")


# ============================================================================
# 9.2 OPTUNA HYPERPARAMETER TUNING — Top 3 Models
# ============================================================================

print(f"\n\n{'=' * 70}")
print(" 9.2 — OPTUNA TUNING (Top 3 Models, 50 Trials Each)")
print("=" * 70)

top3 = baseline_df.head(3)["model"].tolist()
print(f"\n  Tuning: {top3}\n")

tuned_results = {}

for model_name in top3:
    def get_objective(model_name):
        def objective(trial):
            if model_name == "XGBoost":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                    "scale_pos_weight": scale_pw,
                    "eval_metric": "aucpr", "random_state": 42, "use_label_encoder": False,
                }
                m = xgb.XGBClassifier(**params)

            elif model_name == "Random Forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 4, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
                }
                m = RandomForestClassifier(**params)

            elif model_name == "Gradient Boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "random_state": 42,
                }
                m = GradientBoostingClassifier(**params)

            elif model_name == "Extra Trees":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 4, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
                }
                m = ExtraTreesClassifier(**params)

            elif model_name == "Logistic Regression":
                C = trial.suggest_float("C", 0.001, 10, log=True)
                m = LogisticRegression(
                    C=C, class_weight="balanced", max_iter=2000,
                    random_state=42, solver="saga", penalty="l1",
                )

            elif model_name == "AdaBoost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
                    "random_state": 42,
                }
                m = AdaBoostClassifier(**params)

            else:
                return 0.0

            y_proba = cross_val_predict(m, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
            return average_precision_score(y_train, y_proba)

        return objective

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(get_objective(model_name), n_trials=50)
    tuned_results[model_name] = {
        "best_prauc": study.best_value,
        "best_params": study.best_params,
    }
    print(f"  {model_name:30s}  PR-AUC = {study.best_value:.4f}")
    print(f"    Best params: {study.best_params}")


# ============================================================================
# 9.3 ENSEMBLE MODELS
# ============================================================================

print(f"\n\n{'=' * 70}")
print(" 9.3 — ENSEMBLE MODELS")
print("=" * 70)

best_tuned_name = max(tuned_results, key=lambda k: tuned_results[k]["best_prauc"])
best_tuned_params = tuned_results[best_tuned_name]["best_params"]

base_rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
base_xgb = xgb.XGBClassifier(
    n_estimators=200, scale_pos_weight=scale_pw,
    eval_metric="aucpr", random_state=42, use_label_encoder=False,
)
base_gb = GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=4)

ensemble_results = {}

# Voting Ensemble (soft)
print(f"\n  [Voting Ensemble — Soft]")
voting = VotingClassifier(
    estimators=[("rf", base_rf), ("xgb", base_xgb), ("gb", base_gb)],
    voting="soft",
)
y_proba = cross_val_predict(voting, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
res = evaluate_model("Voting (Soft)", y_train, y_proba)
ensemble_results["Voting (Soft)"] = {"metrics": res, "proba": y_proba}
print(f"    PR-AUC={res['PR-AUC']:.4f}  ROC-AUC={res['ROC-AUC']:.4f}  F2={res['Best_F2']:.4f}")

# Stacking Ensemble
print(f"  [Stacking Ensemble]")
stacking = StackingClassifier(
    estimators=[("rf", base_rf), ("xgb", base_xgb), ("gb", base_gb)],
    final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
    cv=5, passthrough=False,
)
y_proba = cross_val_predict(stacking, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
res = evaluate_model("Stacking", y_train, y_proba)
ensemble_results["Stacking"] = {"metrics": res, "proba": y_proba}
print(f"    PR-AUC={res['PR-AUC']:.4f}  ROC-AUC={res['ROC-AUC']:.4f}  F2={res['Best_F2']:.4f}")


# ============================================================================
# 9.4 MODEL COMPARISON & WINNER SELECTION
# ============================================================================

print(f"\n\n{'=' * 70}")
print(" 9.4 — MODEL COMPARISON (All Approaches)")
print("=" * 70)

comparison = []
for r in baseline_results:
    comparison.append({**r, "stage": "baseline"})
for name, info in tuned_results.items():
    comparison.append({
        "model": f"{name} (tuned)", "PR-AUC": info["best_prauc"],
        "ROC-AUC": np.nan, "Best_F2": np.nan, "stage": "tuned",
    })
for name, info in ensemble_results.items():
    comparison.append({**info["metrics"], "stage": "ensemble"})

comp_df = pd.DataFrame(comparison)
comp_numeric = comp_df[comp_df["PR-AUC"].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x) if isinstance(x, float) else True)]
comp_numeric = comp_numeric.sort_values("PR-AUC", ascending=False)

print(f"\n  {'Model':40s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'F2':>8s}  {'Stage':>10s}")
print(f"  {'─' * 80}")
for _, row in comp_numeric.iterrows():
    roc = f"{row['ROC-AUC']:.4f}" if not (isinstance(row['ROC-AUC'], float) and np.isnan(row['ROC-AUC'])) else "   —"
    f2 = f"{row['Best_F2']:.4f}" if not (isinstance(row['Best_F2'], float) and np.isnan(row['Best_F2'])) else "   —"
    marker = " << BEST" if row.name == comp_numeric.index[0] else ""
    print(f"  {row['model']:40s} {row['PR-AUC']:.4f}   {roc:>6s}   {f2:>6s}   {row['stage']:>10s}{marker}")

best_model_name = comp_numeric.iloc[0]["model"]
best_prauc = comp_numeric.iloc[0]["PR-AUC"]
print(f"\n  WINNER: {best_model_name} (PR-AUC = {best_prauc:.4f})")

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
plot_df = comp_numeric[["model", "PR-AUC"]].set_index("model").sort_values("PR-AUC")
colors = ["#e74c3c" if i == len(plot_df) - 1 else "#3498db" for i in range(len(plot_df))]
plot_df.plot.barh(ax=ax, color=colors, legend=False)
ax.set_xlabel("PR-AUC (higher = better)")
ax.set_title("Model Comparison — PR-AUC", fontsize=14, fontweight='bold')
for i, (idx, row) in enumerate(plot_df.iterrows()):
    ax.text(row["PR-AUC"] + 0.002, i, f"{row['PR-AUC']:.4f}", va='center', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_MODEL, "plot_17_model_comparison.png"), dpi=150)
plt.close(fig)


# ============================================================================
# 9.5 RETRAIN BEST MODEL & THRESHOLD OPTIMIZATION
# ============================================================================

print(f"\n\n{'=' * 70}")
print(" 9.5 — RETRAIN BEST MODEL & THRESHOLD OPTIMIZATION")
print("=" * 70)

# Build the final model using the best tuned parameters
if "XGBoost" in best_model_name:
    params = tuned_results.get("XGBoost", {}).get("best_params", {})
    final_model = xgb.XGBClassifier(
        **params, scale_pos_weight=scale_pw,
        eval_metric="aucpr", random_state=42, use_label_encoder=False,
    )
elif "Random Forest" in best_model_name:
    params = tuned_results.get("Random Forest", {}).get("best_params", {})
    final_model = RandomForestClassifier(
        **params, class_weight="balanced", random_state=42, n_jobs=-1,
    )
elif "Gradient" in best_model_name:
    params = tuned_results.get("Gradient Boosting", {}).get("best_params", {})
    final_model = GradientBoostingClassifier(**params, random_state=42)
elif "Voting" in best_model_name:
    final_model = VotingClassifier(
        estimators=[("rf", base_rf), ("xgb", base_xgb), ("gb", base_gb)],
        voting="soft",
    )
elif "Stacking" in best_model_name:
    final_model = StackingClassifier(
        estimators=[("rf", base_rf), ("xgb", base_xgb), ("gb", base_gb)],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        cv=5, passthrough=False,
    )
else:
    params = tuned_results.get("XGBoost", {}).get("best_params", {})
    final_model = xgb.XGBClassifier(
        **params, scale_pos_weight=scale_pw,
        eval_metric="aucpr", random_state=42, use_label_encoder=False,
    )

# Get CV predictions on training set for threshold optimization
y_train_proba = cross_val_predict(final_model, X_train, y_train, cv=CV, method="predict_proba")[:, 1]

# Threshold optimization — maximize F2
print(f"\n  [Threshold Optimization — Maximizing F2 Score]")
thresholds = np.arange(0.15, 0.85, 0.005)
f1_scores, f2_scores, f05_scores = [], [], []
for t in thresholds:
    y_pred_t = (y_train_proba >= t).astype(int)
    f1_scores.append(fbeta_score(y_train, y_pred_t, beta=1))
    f2_scores.append(fbeta_score(y_train, y_pred_t, beta=2))
    f05_scores.append(fbeta_score(y_train, y_pred_t, beta=0.5))

best_f2_idx = np.argmax(f2_scores)
best_threshold = thresholds[best_f2_idx]
print(f"    Best F2 threshold: {best_threshold:.3f}")
print(f"    At threshold: F1={f1_scores[best_f2_idx]:.3f}, F2={f2_scores[best_f2_idx]:.3f}, F0.5={f05_scores[best_f2_idx]:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, f05_scores, label="F0.5 (precision-weighted)", color="#3498db", linewidth=1.5)
ax.plot(thresholds, f1_scores, label="F1 (balanced)", color="#2ecc71", linewidth=1.5)
ax.plot(thresholds, f2_scores, label="F2 (recall-weighted)", color="#e74c3c", linewidth=2)
ax.axvline(best_threshold, color="black", linestyle="--", alpha=0.7, label=f"Best F2 = {best_threshold:.3f}")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("F-beta Scores vs Decision Threshold", fontsize=14, fontweight='bold')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_MODEL, "plot_18_threshold_optimization.png"), dpi=150)
plt.close(fig)

# Retrain on FULL training set
print(f"\n  Retraining final model on full training set...")
final_model.fit(X_train, y_train)


# ============================================================================
# 9.6 FINAL EVALUATION ON TEST SET
# ============================================================================

print(f"\n\n{'=' * 70}")
print(" 9.6 — FINAL EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)

y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

test_prauc = average_precision_score(y_test, y_test_proba)
test_roc = roc_auc_score(y_test, y_test_proba)
test_f2 = fbeta_score(y_test, y_test_pred, beta=2)
test_f1 = f1_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n  Threshold: {best_threshold:.3f}")
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │  TEST SET RESULTS (n={len(y_test)})            │")
print(f"  ├─────────────────────────────────────────┤")
print(f"  │  Accuracy:      {test_acc:.4f}                │")
print(f"  │  Precision:     {test_precision:.4f}                │")
print(f"  │  Recall:        {test_recall:.4f}                │")
print(f"  │  F1 Score:      {test_f1:.4f}                │")
print(f"  │  F2 Score:      {test_f2:.4f}                │")
print(f"  │  PR-AUC:        {test_prauc:.4f}                │")
print(f"  │  ROC-AUC:       {test_roc:.4f}                │")
print(f"  └─────────────────────────────────────────┘")

print(f"\n  Classification Report (threshold={best_threshold:.3f}):\n")
print(classification_report(y_test, y_test_pred, target_names=["No Churn", "Churn"], digits=4))

# Confusion matrix and curves
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred, display_labels=["No Churn", "Churn"], cmap="Blues", ax=axes[0]
)
axes[0].set_title(f"Confusion Matrix (t={best_threshold:.3f})", fontsize=13, fontweight='bold')

pr_disp = PrecisionRecallDisplay.from_predictions(y_test, y_test_proba, ax=axes[1])
pr_disp.line_.set_color("#e74c3c")
axes[1].set_title(f"Precision-Recall Curve (AUC={test_prauc:.3f})", fontsize=13, fontweight='bold')

roc_disp = RocCurveDisplay.from_predictions(y_test, y_test_proba, ax=axes[2])
roc_disp.line_.set_color("#3498db")
axes[2].set_title(f"ROC Curve (AUC={test_roc:.3f})", fontsize=13, fontweight='bold')

fig.suptitle("Final Model Evaluation — Held-Out Test Set", fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_MODEL, "plot_19_final_evaluation.png"), dpi=150, bbox_inches='tight')
plt.close(fig)


# Also evaluate at default threshold (0.5) for comparison
y_test_pred_default = (y_test_proba >= 0.5).astype(int)
print(f"\n  Comparison — Default threshold (0.5) vs Optimized ({best_threshold:.3f}):")
print(f"    {'Metric':15s}  {'t=0.5':>8s}  {'t=' + f'{best_threshold:.3f}':>8s}  {'Change':>10s}")
print(f"    {'─' * 50}")
for metric_name, metric_fn in [("Recall", recall_score), ("Precision", precision_score),
                                 ("F1", f1_score)]:
    v1 = metric_fn(y_test, y_test_pred_default)
    v2 = metric_fn(y_test, y_test_pred)
    diff = v2 - v1
    print(f"    {metric_name:15s}  {v1:.4f}    {v2:.4f}    {diff:+.4f}")
f2_default = fbeta_score(y_test, y_test_pred_default, beta=2)
diff_f2 = test_f2 - f2_default
print(f"    {'F2':15s}  {f2_default:.4f}    {test_f2:.4f}    {diff_f2:+.4f}")


# ============================================================================
# 9.7 SHAP ANALYSIS
# ============================================================================

print(f"\n\n{'=' * 70}")
print(" 9.7 — SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)

# For SHAP, use the underlying model (not ensemble wrappers)
# If the winner was an ensemble, retrain the best tuned single model for SHAP
if any(keyword in best_model_name for keyword in ["Voting", "Stacking"]):
    print(f"\n  Winner was an ensemble — using best tuned single model for SHAP...")
    shap_model_name = max(tuned_results, key=lambda k: tuned_results[k]["best_prauc"])
    shap_params = tuned_results[shap_model_name]["best_params"]
    if "XGBoost" in shap_model_name:
        shap_model = xgb.XGBClassifier(
            **shap_params, scale_pos_weight=scale_pw,
            eval_metric="aucpr", random_state=42, use_label_encoder=False,
        )
    elif "Random Forest" in shap_model_name:
        shap_model = RandomForestClassifier(
            **shap_params, class_weight="balanced", random_state=42, n_jobs=-1,
        )
    elif "Gradient" in shap_model_name:
        shap_model = GradientBoostingClassifier(**shap_params, random_state=42)
    else:
        shap_model = xgb.XGBClassifier(
            n_estimators=200, scale_pos_weight=scale_pw,
            eval_metric="aucpr", random_state=42, use_label_encoder=False,
        )
    shap_model.fit(X_train, y_train)
    print(f"  Using: {shap_model_name} for SHAP analysis")
else:
    shap_model = final_model
    shap_model_name = best_model_name
    print(f"\n  Using winner model for SHAP: {shap_model_name}")


# --- 9.7.1 SHAP Values Computation ---
print(f"\n  Computing SHAP values...")
if isinstance(shap_model, xgb.XGBClassifier):
    explainer = shap.TreeExplainer(shap_model)
elif isinstance(shap_model, (RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier)):
    explainer = shap.TreeExplainer(shap_model)
else:
    explainer = shap.LinearExplainer(shap_model, X_train)

shap_values = explainer.shap_values(X_test)

# Handle different SHAP value formats
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # class 1 (churn)
else:
    shap_vals = shap_values

print(f"  SHAP values shape: {shap_vals.shape}")


# --- 9.7.2 Feature Importance (Mean |SHAP|) ---
print(f"\n  [Feature Importance — Mean |SHAP| Values]")

feature_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean_SHAP': np.abs(shap_vals).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)

print(f"\n  {'Rank':>4s}  {'Feature':40s}  {'Mean |SHAP|':>12s}")
print(f"  {'─' * 62}")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
    print(f"  {i+1:4d}  {row['Feature']:40s}  {row['Mean_SHAP']:.6f}")


# --- 9.7.3 SHAP Summary Plot (Beeswarm) ---
print(f"\n  Generating SHAP plots...")

fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_vals, X_test, max_display=20, show=False)
plt.title("SHAP Summary Plot — Feature Impact on Churn Prediction", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_SHAP, "plot_20_shap_summary.png"), dpi=150, bbox_inches='tight')
plt.close()


# --- 9.7.4 SHAP Bar Plot (Global Importance) ---
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, X_test, plot_type="bar", max_display=20, show=False)
plt.title("SHAP Feature Importance — Mean |SHAP Value|", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_SHAP, "plot_21_shap_bar.png"), dpi=150, bbox_inches='tight')
plt.close()


# --- 9.7.5 SHAP Dependence Plots (Top 4 Features) ---
top4_features = feature_importance.head(4)['Feature'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feat in enumerate(top4_features):
    shap.dependence_plot(
        feat, shap_vals, X_test,
        ax=axes[i], show=False,
    )
    axes[i].set_title(f"SHAP Dependence: {feat}", fontsize=12, fontweight='bold')

fig.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_SHAP, "plot_22_shap_dependence.png"), dpi=150, bbox_inches='tight')
plt.close(fig)


# --- 9.7.6 SHAP Force Plots — Individual Predictions ---
print(f"\n  [Individual Prediction Explanations]")

# Find a high-confidence churn prediction and a no-churn prediction
churn_idx = np.where((y_test_pred == 1) & (y_test_proba > 0.8))[0]
nochurn_idx = np.where((y_test_pred == 0) & (y_test_proba < 0.2))[0]

if len(churn_idx) > 0:
    idx = churn_idx[0]
    print(f"\n  High-confidence CHURN prediction (index={idx}, prob={y_test_proba[idx]:.3f}):")
    top_shap = pd.Series(shap_vals[idx], index=X_test.columns).abs().sort_values(ascending=False).head(5)
    for feat_name, val in top_shap.items():
        actual_shap = shap_vals[idx][list(X_test.columns).index(feat_name)]
        direction = "pushes TOWARD churn" if actual_shap > 0 else "pushes AWAY from churn"
        feat_val = X_test.iloc[idx][feat_name]
        print(f"    {feat_name} = {feat_val:.2f}  →  SHAP = {actual_shap:+.4f}  ({direction})")

if len(nochurn_idx) > 0:
    idx = nochurn_idx[0]
    print(f"\n  High-confidence NO-CHURN prediction (index={idx}, prob={y_test_proba[idx]:.3f}):")
    top_shap = pd.Series(shap_vals[idx], index=X_test.columns).abs().sort_values(ascending=False).head(5)
    for feat_name, val in top_shap.items():
        actual_shap = shap_vals[idx][list(X_test.columns).index(feat_name)]
        direction = "pushes TOWARD churn" if actual_shap > 0 else "pushes AWAY from churn"
        feat_val = X_test.iloc[idx][feat_name]
        print(f"    {feat_name} = {feat_val:.2f}  →  SHAP = {actual_shap:+.4f}  ({direction})")


# --- 9.7.7 SHAP Waterfall — Single Prediction Decomposition ---
if len(churn_idx) > 0:
    idx = churn_idx[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    explanation = shap.Explanation(
        values=shap_vals[idx],
        base_values=expected_value,
        data=X_test.iloc[idx].values,
        feature_names=list(X_test.columns)
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — High-Confidence Churn (prob={y_test_proba[churn_idx[0]]:.3f})",
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_SHAP, "plot_23_shap_waterfall_churn.png"), dpi=150, bbox_inches='tight')
    plt.close()

if len(nochurn_idx) > 0:
    idx = nochurn_idx[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    explanation = shap.Explanation(
        values=shap_vals[idx],
        base_values=expected_value,
        data=X_test.iloc[idx].values,
        feature_names=list(X_test.columns)
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — High-Confidence No-Churn (prob={y_test_proba[nochurn_idx[0]]:.3f})",
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_SHAP, "plot_24_shap_waterfall_nochurn.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 9.8 SAVE FINAL MODEL & PREDICTIONS
# ============================================================================

print(f"\n\n{'=' * 70}")
print(" 9.8 — SAVING ARTIFACTS")
print("=" * 70)

# Save model
model_path = os.path.join(ARTIFACTS_DIR, "best_churn_model.joblib")
joblib.dump({
    "model": final_model,
    "threshold": best_threshold,
    "model_name": best_model_name,
    "test_prauc": test_prauc,
    "test_roc_auc": test_roc,
    "test_f2": test_f2,
    "features": list(X_train.columns),
}, model_path)
print(f"  Saved model → {model_path}")

# Save predictions
predictions_df = X_test.copy()
predictions_df["actual_churn"] = y_test.values
predictions_df["predicted_churn"] = y_test_pred
predictions_df["churn_probability"] = y_test_proba
predictions_df.to_csv(os.path.join(REPORTS_MODEL, "test_predictions.csv"), index=False)
print(f"  Saved predictions → {os.path.join(REPORTS_MODEL, 'test_predictions.csv')}")

# Save feature importance
feature_importance.to_csv(os.path.join(REPORTS_SHAP, "shap_feature_importance.csv"), index=False)
print(f"  Saved SHAP importance → {os.path.join(REPORTS_SHAP, 'shap_feature_importance.csv')}")

# Save SHAP values
shap_df = pd.DataFrame(shap_vals, columns=X_test.columns)
shap_df.to_csv(os.path.join(REPORTS_SHAP, "shap_values.csv"), index=False)
print(f"  Saved SHAP values → {os.path.join(REPORTS_SHAP, 'shap_values.csv')}")


# ============================================================================
# STEP 9 SUMMARY
# ============================================================================

print(f"\n\n{'=' * 70}")
print(" STEP 9 — COMPLETE SUMMARY")
print("=" * 70)
print(f"""
  MODEL PIPELINE:
    7 baseline models → Optuna tuning (top 3, 50 trials) → 2 ensembles
    Winner: {best_model_name}

  TEST SET PERFORMANCE (threshold = {best_threshold:.3f}):
    PR-AUC:    {test_prauc:.4f}
    ROC-AUC:   {test_roc:.4f}
    F2 Score:  {test_f2:.4f}
    Recall:    {test_recall:.4f}  (of actual churners caught)
    Precision: {test_precision:.4f}  (of predicted churners correct)
    Accuracy:  {test_acc:.4f}

  SHAP TOP 5 FEATURES:
""")
for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
    print(f"    {i+1}. {row['Feature']} (mean |SHAP| = {row['Mean_SHAP']:.4f})")

print(f"""
  OUTPUTS:
    reports/modelling/plot_17_model_comparison.png
    reports/modelling/plot_18_threshold_optimization.png
    reports/modelling/plot_19_final_evaluation.png
    reports/shap/plot_20_shap_summary.png
    reports/shap/plot_21_shap_bar.png
    reports/shap/plot_22_shap_dependence.png
    reports/shap/plot_23_shap_waterfall_churn.png
    reports/shap/plot_24_shap_waterfall_nochurn.png
    reports/modelling/test_predictions.csv
    reports/shap/shap_feature_importance.csv
    reports/shap/shap_values.csv
    artifacts/best_churn_model.joblib
""")
print("=" * 70)
