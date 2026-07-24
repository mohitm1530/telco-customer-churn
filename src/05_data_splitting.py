"""
=============================================================================
 Telco Customer Churn — Data Preprocessing Pipeline
 Step 6: Data Splitting
=============================================================================

Data Splitting = dividing data into SEPARATE sets for training and evaluation.

WHY split at all?
  If you train AND evaluate on the same data, the model just memorizes it.
  It scores 100% on data it has seen, but fails on new customers.
  This is called OVERFITTING.

  Splitting creates data the model has NEVER seen during training, giving
  you an honest estimate of how it will perform in production.

THE CRITICAL RULE:
  The test set must be a "time capsule" — NEVER touched during training,
  feature engineering, scaling, or balancing. Any information that leaks
  from test → train gives you falsely optimistic results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(ROOT_DIR, 'data', 'processed')
DATA_SPLITS = os.path.join(ROOT_DIR, 'data', 'splits')
ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'artifacts')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports', 'splitting')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ============================================================================
# LOAD SELECTED + ENCODED DATA
# ============================================================================

df = pd.read_csv(os.path.join(DATA_PROCESSED, 'telco_churn_selected.csv'))
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

target = 'Churn Value'
X = df.drop(columns=[target])
y = df[target]

print(f"Features (X): {X.shape}")
print(f"Target (y):   {y.shape}")
print(f"Target balance: {dict(y.value_counts())} → {y.mean()*100:.1f}% churn\n")


# ============================================================================
# 6.1 TRAIN-TEST SPLIT
# ============================================================================
#
# WHAT: Divide data into training set (~80%) and test set (~20%).
#
# KEY DECISIONS:
#
# ── 1. Split Ratio: WHY 80/20 (not 70/30 or 90/10)?  ──
#
#   Option A: 70/30
#     → More test data = more reliable evaluation.
#     → BUT: less training data = model learns less.
#     → Used when: dataset is LARGE (100K+ rows) and you can afford to
#       hold out 30% without starving the model.
#
#   Option B: 80/20 (our choice)
#     → Industry standard balance between training power and evaluation
#       reliability.
#     → With 7,043 rows, 20% = ~1,409 test samples. That's enough for
#       stable metrics (at least 300+ per class after stratification).
#     → Used when: medium-sized datasets (1K-100K rows).
#
#   Option C: 90/10
#     → Maximizes training data, but test set is small and noisy.
#     → With 7,043 rows, 10% = ~704 test samples. Metrics would have
#       high variance — a few lucky/unlucky samples swing the results.
#     → Used when: dataset is SMALL (<1K rows) and every training sample
#       matters.
#
#
# ── 2. Stratification: WHY stratify=y? ──
#
#   Our target is IMBALANCED: 73.5% class 0, 26.5% class 1.
#
#   WITHOUT stratification (random split):
#     → By random chance, the test set might get 30% churn or 22% churn.
#     → This means your evaluation is on a DIFFERENT distribution than
#       what the model trained on → unreliable metrics.
#     → Worst case: very few minority samples in test set → unstable
#       recall/F1 scores.
#
#   WITH stratification:
#     → GUARANTEES that train and test have the SAME class proportions
#       (26.5% churn in both).
#     → Evaluation reflects real-world performance fairly.
#     → Especially critical when minority class is <30% (our case).
#
#   WHY NOT stratify on other features too?
#     → Multi-feature stratification is possible but complex and usually
#       unnecessary. The target is the variable whose distribution matters
#       most for evaluation. If you also stratify on Contract type AND
#       Internet Service, you quickly run out of samples in rare combinations.
#
#
# ── 3. Random State: WHY fix it? ──
#
#   random_state=42 makes the split REPRODUCIBLE. Same code → same split
#   every time. Without it, each run produces different train/test sets,
#   making it impossible to compare experiments fairly.
#   The specific number (42) doesn't matter — it's just a convention.
#
#
# ── 4. Shuffle: WHY shuffle=True? ──
#
#   If the data is ordered (e.g., first 3000 rows are churned, last 4000
#   are not), a non-shuffled split would put all churned customers in
#   training and none in test. Shuffling randomizes the order first.
#   With our data, rows appear mixed, but shuffling is a safety measure.
# ============================================================================

print("6.1 — Train-Test Split (80/20, Stratified)")
print("─" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
    shuffle=True
)

print(f"  Training set: {X_train.shape[0]} rows ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test set:     {X_test.shape[0]} rows ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"\n  Stratification check (churn rate should be ~26.5% in both):")
print(f"    Full dataset: {y.mean()*100:.2f}%")
print(f"    Training set: {y_train.mean()*100:.2f}%")
print(f"    Test set:     {y_test.mean()*100:.2f}%")

# Verify stratification succeeded
assert abs(y_train.mean() - y_test.mean()) < 0.01, "Stratification failed!"
print("    ✓ Stratification verified — proportions match.\n")


# ============================================================================
# 6.2 FEATURE SCALING
# ============================================================================
#
# WHAT: Transform numerical features so they're on comparable scales.
#
# WHY scale at all?
#   Our numerical features have vastly different ranges:
#     - Tenure Months: 0 — 72
#     - Monthly Charges: 18.25 — 118.75
#     - Num Addon Services: 0 — 6
#   Algorithms that use DISTANCES (KNN, SVM) or GRADIENTS (Logistic
#   Regression, Neural Networks) are heavily biased toward large-scale
#   features. Monthly Charges would dominate just because it has bigger
#   numbers, not because it's more important.
#
#   Tree-based models (Random Forest, XGBoost) DON'T need scaling because
#   they split on thresholds, not distances. But scaling doesn't hurt them
#   either, so we scale for generality.
#
# WHICH scaler?
#
#   Option A: StandardScaler (Z-score normalization)
#     → Transforms to mean=0, std=1:  z = (x - mean) / std
#     → Works best when: features are roughly normally distributed
#     → Robust to: moderate outliers (outliers shift mean/std but don't
#       break the transformation)
#     → OUR CHOICE because: Logistic Regression (common for churn) assumes
#       features are centered. Also works well for SVM and neural nets.
#
#   Option B: MinMaxScaler
#     → Transforms to [0, 1] range:  x' = (x - min) / (max - min)
#     → Works best when: you need bounded values (e.g., neural network
#       inputs), or features are uniformly distributed
#     → FRAGILE to outliers: a single extreme value compresses all other
#       values into a tiny range
#     → Not our choice because: if a new customer has tenure=100 (beyond
#       our max of 72), MinMax gives a value >1, breaking the [0,1] contract.
#
#   Option C: RobustScaler
#     → Uses median and IQR instead of mean and std
#     → Most resistant to outliers
#     → Good when: data has significant outliers (ours doesn't — EDA
#       confirmed all values are within business ranges)
#
#   Option D: No scaling
#     → Only valid if using ONLY tree-based models
#     → We keep our options open for any algorithm, so we scale.
#
#
# CRITICAL: fit on TRAINING data, transform BOTH
#
#   scaler.fit(X_train)       → learns mean/std from training data ONLY
#   scaler.transform(X_train) → applies to training data
#   scaler.transform(X_test)  → applies SAME mean/std to test data
#
#   WHY NOT fit on the full dataset?
#   → fit() computes statistics (mean, std). If test data is included,
#     the scaler "knows" the test distribution — this is DATA LEAKAGE.
#   → The model indirectly learns about test data through the scaling
#     parameters, giving falsely optimistic results.
#
#   WHY NOT fit separately on train and test?
#   → Train and test would be on DIFFERENT scales. A tenure of 36 months
#     might map to z=0.15 in training but z=0.20 in test. The model would
#     interpret the same real value differently — inconsistent predictions.
# ============================================================================

print("6.2 — Feature Scaling (StandardScaler)")
print("─" * 60)

# Identify which columns to scale:
#   - Continuous: Tenure Months, Monthly Charges (wide ranges)
#   - Ordinal with meaningful range: Num Addon Services (0-6), Tenure Group (0-3),
#     Charges Tier (0-2) — these have ranges that affect distance-based algorithms
#
# Which columns NOT to scale:
#   - Binary (0/1): scaling shifts them off 0/1 → harder to interpret,
#     and they're already on a uniform scale
#   - One-hot encoded: already 0/1 binary indicators
#
# WHY scale ordinal features too?
#   Num Addon Services ranges 0-6 while binary features are 0-1.
#   In distance-based models (KNN, SVM), this 6x range difference makes
#   add-ons dominate over binary features. StandardScaler puts them all
#   on the same scale.
numerical_to_scale = ['Tenure Months', 'Monthly Charges', 'Num Addon Services',
                      'Tenure Group', 'Charges Tier']

# Binary and one-hot columns stay as-is
already_encoded = [c for c in X_train.columns if c not in numerical_to_scale]

print(f"  Columns to scale: {numerical_to_scale}")
print(f"  Columns to keep as-is: {len(already_encoded)} encoded features\n")

scaler = StandardScaler()

# fit on TRAIN only, transform both
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_to_scale] = scaler.fit_transform(X_train[numerical_to_scale])
X_test_scaled[numerical_to_scale] = scaler.transform(X_test[numerical_to_scale])

print("  Scaling statistics (learned from TRAINING set only):")
for col, mean, std in zip(numerical_to_scale, scaler.mean_, scaler.scale_):
    print(f"    {col}: mean={mean:.2f}, std={std:.2f}")

print(f"\n  After scaling — Training set sample:")
print(X_train_scaled[numerical_to_scale].describe().round(3).to_string())

print(f"\n  After scaling — Test set sample:")
print(X_test_scaled[numerical_to_scale].describe().round(3).to_string())
print()


# ============================================================================
# 6.3 VISUALIZATION — Scaling Effect
# ============================================================================

# Show the 2 most illustrative columns (continuous ones with widest range)
plot_cols = ['Tenure Months', 'Monthly Charges']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for i, col in enumerate(plot_cols):
    # Before scaling
    axes[0, i].hist(X_train[col], bins=30, alpha=0.6, color='steelblue', edgecolor='black', label='Train')
    axes[0, i].hist(X_test[col], bins=30, alpha=0.4, color='coral', edgecolor='black', label='Test')
    axes[0, i].set_title(f'{col} — Before Scaling', fontsize=13, fontweight='bold')
    axes[0, i].legend()
    axes[0, i].set_ylabel('Count')

    # After scaling
    axes[1, i].hist(X_train_scaled[col], bins=30, alpha=0.6, color='steelblue', edgecolor='black', label='Train')
    axes[1, i].hist(X_test_scaled[col], bins=30, alpha=0.4, color='coral', edgecolor='black', label='Test')
    axes[1, i].set_title(f'{col} — After Scaling (z-score)', fontsize=13, fontweight='bold')
    axes[1, i].legend()
    axes[1, i].set_ylabel('Count')
    axes[1, i].set_xlabel('Standardized Value')

plt.suptitle('Feature Scaling: Before vs After\n(Shape stays the same, scale changes to mean=0, std=1)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_15_scaling_effect.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# 6.4 SAVE SPLIT DATA
# ============================================================================

import joblib

X_train_scaled.to_csv(os.path.join(DATA_SPLITS, 'X_train.csv'), index=False)
X_test_scaled.to_csv(os.path.join(DATA_SPLITS, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(DATA_SPLITS, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(DATA_SPLITS, 'y_test.csv'), index=False)

# Save the scaler for production use — new data must be scaled with the
# SAME mean/std learned from training data
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))

print("  Saved: data/splits/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("  Saved: artifacts/scaler.joblib (for transforming new data in production)\n")


# ============================================================================
# 6.5 WHY NOT OTHER SPLIT STRATEGIES?
# ============================================================================

print("=" * 65)
print(" 6.5 — WHY NOT OTHER SPLIT STRATEGIES?")
print("=" * 65)
print("""
 Strategy              | When to Use              | Why NOT Here
 ----------------------|--------------------------|---------------------------
 K-Fold Cross-Val      | Small datasets (<2K rows)| We have 7K rows — enough
                       | Need robust estimates    | for a reliable single split.
                       |                          | We'll use K-Fold during
                       |                          | model tuning, not here.
                       |                          |
 Leave-One-Out (LOO)   | Very small (<200 rows)   | Way too expensive for 7K
                       |                          | rows (7K model trainings!)
                       |                          |
 Time-Based Split      | When data has temporal   | Our data isn't time-ordered.
                       | ordering (stock prices,  | If it were, we'd use the
                       | logs with timestamps)    | last N months as test set.
                       |                          |
 Train/Val/Test        | Deep learning / complex  | For traditional ML with
 (3-way split)         | hyperparameter tuning    | GridSearchCV, the CV folds
                       |                          | serve as validation sets.
                       |                          | A separate val set would
                       |                          | waste training data.
""")


# ============================================================================
# STEP 6 SUMMARY
# ============================================================================

print("=" * 65)
print(" STEP 6 — DATA SPLITTING SUMMARY")
print("=" * 65)
print(f"""
 What We Did                  | Why This Approach
 -----------------------------|---------------------------------------
 80/20 stratified split       | Standard ratio; stratify=y preserves
                              | 26.5% churn rate in both sets
 StandardScaler on numericals | Centers features (mean=0, std=1) for
                              | distance/gradient-based algorithms
 fit() on TRAIN only          | Prevents test data leaking into scaling
                              | parameters (data leakage prevention)
 Saved 4 separate files       | Clean separation for next steps

 Training: {X_train_scaled.shape[0]} rows × {X_train_scaled.shape[1]} features
 Test:     {X_test_scaled.shape[0]} rows × {X_test_scaled.shape[1]} features
 Target:   {y_train.mean()*100:.1f}% churn (both sets)

 NEXT: Step 7 — Handling Class Imbalance
""")
print("=" * 65)
