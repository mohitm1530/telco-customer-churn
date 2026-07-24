"""
=============================================================================
 Telco Customer Churn — Data Preprocessing Pipeline
 Step 7: Handling Class Imbalance
=============================================================================

Class Imbalance = when one class has far more samples than the other.
Our dataset: 73.5% No-Churn vs 26.5% Churn.

WHY is this a problem?
  A model trained on imbalanced data learns a shortcut: "just predict the
  majority class for everyone." This gives 73.5% accuracy but ZERO ability
  to detect churners — the very thing we're trying to predict.

  Metrics like accuracy become MISLEADING. What matters is:
  - Recall (of churners): What % of actual churners did we catch?
  - Precision: Of the customers we flagged as churners, how many truly are?
  - F1 Score: Harmonic mean of precision and recall

THE CRITICAL RULE:
  ALL balancing must happen on the TRAINING set ONLY. NEVER touch the
  test set. The test set must reflect the real-world distribution so
  your evaluation is honest.

  WHY? If you SMOTE the test set too:
  - You're evaluating on SYNTHETIC data, not real customers
  - Synthetic test samples may overlap with training data → leakage
  - Your metrics look great on fake data, fail on real customers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
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
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports', 'imbalance')
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# LOAD SPLIT DATA
# ============================================================================

X_train = pd.read_csv(os.path.join(DATA_SPLITS, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(DATA_SPLITS, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_SPLITS, 'y_train.csv')).squeeze()
y_test = pd.read_csv(os.path.join(DATA_SPLITS, 'y_test.csv')).squeeze()

print(f"Training set: {X_train.shape[0]} rows × {X_train.shape[1]} features")
print(f"Test set:     {X_test.shape[0]} rows × {X_test.shape[1]} features")
print(f"\nTraining target distribution:")
print(f"  {dict(y_train.value_counts())}")
print(f"  Churn rate: {y_train.mean()*100:.1f}%")
print(f"  Imbalance ratio: {y_train.value_counts()[0] / y_train.value_counts()[1]:.1f}:1\n")


# ============================================================================
# 7.1 UNDERSTAND THE OPTIONS
# ============================================================================
#
# There are 4 main approaches to handling class imbalance.
# Each has trade-offs. We'll implement ALL of them and compare:
#
# ┌────────────────────┬───────────────┬──────────────┬──────────────────────┐
# │ Method             │ Changes Data? │ Changes Model│ Key Trade-off        │
# ├────────────────────┼───────────────┼──────────────┼──────────────────────┤
# │ 1. Random Over-    │ YES (copies   │ NO           │ Simple but risks     │
# │    sampling        │ minority rows)│              │ overfitting to exact │
# │                    │               │              │ copies of minority   │
# ├────────────────────┼───────────────┼──────────────┼──────────────────────┤
# │ 2. SMOTE           │ YES (creates  │ NO           │ Avoids exact copies  │
# │                    │ SYNTHETIC     │              │ but assumes feature  │
# │                    │ minority rows)│              │ space is continuous   │
# ├────────────────────┼───────────────┼──────────────┼──────────────────────┤
# │ 3. Random Under-   │ YES (removes  │ NO           │ Loses majority data  │
# │    sampling        │ majority rows)│              │ → less training info │
# ├────────────────────┼───────────────┼──────────────┼──────────────────────┤
# │ 4. Class Weights   │ NO            │ YES (adjusts │ No data change, no   │
# │                    │               │ loss function│ leakage risk, but    │
# │                    │               │ penalties)   │ not all models       │
# │                    │               │              │ support it           │
# └────────────────────┴───────────────┴──────────────┴──────────────────────┘
# ============================================================================

print("=" * 65)
print(" 7.1 — IMPLEMENTING ALL 4 BALANCING METHODS")
print("=" * 65)


# ============================================================================
# 7.2 METHOD 1: RANDOM OVERSAMPLING
# ============================================================================
#
# WHAT: Randomly duplicates minority class samples until balanced.
#
# HOW: Picks random churned customers and copies them exactly.
#   Before: 4139 no-churn, 1495 churn
#   After:  4139 no-churn, 4139 churn (churn copies were duplicated)
#
# PROS:
#   - Dead simple — no assumptions about data distribution
#   - Works with any data type (numerical, categorical, mixed)
#   - No synthetic data means no "impossible" data points
#
# CONS:
#   - Exact copies → model may MEMORIZE specific churned customers
#     instead of learning general churn patterns
#   - Overfitting risk: if the same 5 churned customers appear 3x each,
#     the model over-weights their specific feature values
#
# WHEN TO USE:
#   - As a baseline to compare against SMOTE
#   - When features are mostly categorical (SMOTE struggles with binary)
#   - When you want the simplest possible approach
# ============================================================================

print("\n7.2 — Random Oversampling")
print("─" * 60)

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

print(f"  Before: {dict(Counter(y_train))}")
print(f"  After:  {dict(Counter(y_train_ros))}")
print(f"  Rows added: {len(X_train_ros) - len(X_train)} (exact copies of minority)")
print()


# ============================================================================
# 7.3 METHOD 2: SMOTE (Synthetic Minority Oversampling Technique)
# ============================================================================
#
# WHAT: Creates SYNTHETIC minority samples by interpolating between
#   existing minority samples and their nearest neighbors.
#
# HOW IT WORKS (step by step):
#   1. Pick a minority sample (a churned customer)
#   2. Find its K nearest neighbors (also churned customers)
#   3. Pick one neighbor randomly
#   4. Create a NEW synthetic sample on the line between them:
#      new_sample = original + random_fraction × (neighbor - original)
#
#   Example: Customer A has tenure=5, charges=$80
#            Neighbor B has tenure=9, charges=$90
#            Synthetic: tenure=5+0.6×(9-5)=7.4, charges=$80+0.6×($90-$80)=$86
#
# WHY SMOTE OVER Random Oversampling?
#   - Random oversampling creates EXACT copies → overfitting to specific points
#   - SMOTE creates NEW points in the NEIGHBORHOOD of existing ones
#   - This fills in the feature space, giving the model a richer view of
#     what "churned" looks like across a range of feature values
#   - Less overfitting because the model sees variety, not repetition
#
# LIMITATIONS OF SMOTE:
#   - Assumes features are CONTINUOUS. For binary/one-hot features (most
#     of ours!), interpolation creates fractional values like 0.4 for a
#     column that should be 0 or 1.
#   - Can create "impossible" samples if minority samples are near the
#     majority boundary (synthetic points land in majority territory)
#   - Requires careful k_neighbors setting: too small = overfitting to
#     local clusters; too large = overly smooth synthetic samples
#
# WHY WE STILL USE IT:
#   Despite having many binary features, SMOTE is the industry standard
#   for imbalanced classification. The fractional values in binary columns
#   are tolerable because tree-based models split on thresholds (0.5
#   separates 0 from 1 correctly), and linear models treat them as
#   soft probabilities (which is actually reasonable).
#
# PARAMETERS:
#   k_neighbors=5: Default. Each synthetic sample draws from 5 nearest
#   minority neighbors. Balanced between local detail and generalization.
# ============================================================================

print("7.3 — SMOTE (Synthetic Minority Oversampling)")
print("─" * 60)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"  Before: {dict(Counter(y_train))}")
print(f"  After:  {dict(Counter(y_train_smote))}")
print(f"  Synthetic rows created: {len(X_train_smote) - len(X_train)}")

# Show a synthetic sample vs original
print(f"\n  Example — Synthetic vs Original samples (first 5 features):")
original_churned = X_train[y_train == 1].iloc[:2]
synthetic_new = X_train_smote.iloc[len(X_train):][:2]  # first 2 synthetic rows
compare_cols = X_train.columns[:5].tolist()
print(f"    Original churned customers:")
print(f"    {original_churned[compare_cols].to_string()}")
print(f"    Synthetic (SMOTE-generated):")
print(f"    {synthetic_new[compare_cols].to_string()}")
print("    → Notice: synthetic values are INTERPOLATIONS, not exact copies.")
print()


# ============================================================================
# 7.4 METHOD 3: RANDOM UNDERSAMPLING
# ============================================================================
#
# WHAT: Randomly removes majority class samples until balanced.
#
# HOW: Deletes random no-churn customers until their count matches churned.
#   Before: 4139 no-churn, 1495 churn
#   After:  1495 no-churn, 1495 churn (majority was shrunk)
#
# PROS:
#   - Reduces training time (smaller dataset)
#   - No synthetic data, no overfitting risk from copies
#   - Forces model to focus on the minority class
#
# CONS:
#   - LOSES DATA. You throw away ~64% of your majority class.
#   - The discarded no-churn customers might contain unique patterns
#     that help distinguish loyal customers from at-risk ones.
#   - Only viable when you have LOTS of data (100K+ rows). With ~4K
#     majority samples, losing 2,600 is significant.
#
# WHEN TO USE:
#   - Large datasets where training time is a bottleneck
#   - Combined with oversampling (e.g., SMOTE + undersampling together)
#   - When majority class is very homogeneous (all similar patterns)
#
# WHY NOT our primary choice:
#   We only have 4,139 no-churn samples. Dropping to 1,495 loses
#   valuable information about loyal customer patterns. SMOTE adds
#   data instead of removing it — better for our dataset size.
# ============================================================================

print("7.4 — Random Undersampling")
print("─" * 60)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"  Before: {dict(Counter(y_train))}")
print(f"  After:  {dict(Counter(y_train_rus))}")
print(f"  Rows REMOVED: {len(X_train) - len(X_train_rus)} (majority class data lost)")
print(f"  Dataset shrunk from {len(X_train)} → {len(X_train_rus)} rows ({len(X_train_rus)/len(X_train)*100:.0f}% of original)")
print()


# ============================================================================
# 7.5 METHOD 4: CLASS WEIGHTS
# ============================================================================
#
# WHAT: Instead of changing the data, tell the model to PENALIZE mistakes
#   on the minority class more heavily.
#
# HOW: Assigns a weight to each class inversely proportional to its frequency.
#   weight_i = n_samples / (n_classes × n_samples_i)
#   - Class 0 (no-churn, 4139 samples): weight = 5634 / (2 × 4139) = 0.68
#   - Class 1 (churn, 1495 samples):    weight = 5634 / (2 × 1495) = 1.88
#
#   This means misclassifying a churned customer costs 2.8x more than
#   misclassifying a non-churned one. The model learns to pay attention
#   to the minority class.
#
# WHY class weights OVER resampling?
#   PROS:
#     - No data modification → NO leakage risk, no synthetic artifacts
#     - No increase in dataset size → same training time
#     - Elegant: adjusts the OBJECTIVE FUNCTION, not the data
#     - Works with any model that accepts class_weight parameter
#       (Logistic Regression, SVM, Random Forest, XGBoost)
#
#   CONS:
#     - Not all algorithms support it (KNN doesn't)
#     - Doesn't help if the minority class is too small to learn from
#       (weights can't substitute for insufficient examples)
#     - Less effective than SMOTE when minority class has complex,
#       multi-cluster patterns (weights just up-weight existing points,
#       they don't create new ones to fill gaps)
#
# WHY THIS IS OFTEN THE BEST FIRST CHOICE:
#   Class weights are the safest approach. No data modification = no
#   side effects. Many experienced practitioners start with class weights
#   and only move to SMOTE if weights don't produce enough recall.
# ============================================================================

print("7.5 — Class Weights")
print("─" * 60)

class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
weight_dict = {0: round(class_weights[0], 4), 1: round(class_weights[1], 4)}

print(f"  Computed class weights: {weight_dict}")
print(f"  Interpretation: misclassifying a churned customer costs {weight_dict[1]/weight_dict[0]:.1f}x more")
print(f"  Usage: pass class_weight='balanced' or class_weight={weight_dict}")
print(f"         to LogisticRegression, RandomForest, SVM, etc.")
print(f"\n  NOTE: Class weights don't change the data — they modify the")
print(f"  model's loss function. So X_train and y_train remain unchanged.")
print()


# ============================================================================
# 7.6 COMPARISON — All 4 Methods Side by Side
# ============================================================================

print("=" * 65)
print(" 7.6 — COMPARISON OF ALL 4 METHODS")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

datasets = [
    ('Original\n(Imbalanced)', y_train, len(X_train)),
    ('Random\nOversampling', y_train_ros, len(X_train_ros)),
    ('SMOTE', y_train_smote, len(X_train_smote)),
    ('Random\nUndersampling', y_train_rus, len(X_train_rus)),
    ('Class Weights\n(data unchanged)', y_train, len(X_train)),
]

colors = {'No Churn (0)': '#2ecc71', 'Churn (1)': '#e74c3c'}

for i, (name, y_data, total) in enumerate(datasets):
    row, col = divmod(i, 3)
    counts = pd.Series(y_data).value_counts().sort_index()

    axes[row, col].bar(['No Churn (0)', 'Churn (1)'], counts.values,
                       color=['#2ecc71', '#e74c3c'], edgecolor='black')
    for j, (label, count) in enumerate(zip(['No Churn', 'Churn'], counts.values)):
        axes[row, col].text(j, count + 30, f'{count}\n({count/total*100:.0f}%)',
                           ha='center', fontweight='bold')
    axes[row, col].set_title(name, fontsize=13, fontweight='bold')
    axes[row, col].set_ylabel('Count')
    axes[row, col].set_ylim(0, max(counts.values) * 1.2)

# Use last subplot for summary text
axes[1, 2].axis('off')
summary_text = """
RECOMMENDATION:

Primary:   Class Weights
  → Safest, no data modification,
  → works with most algorithms

Backup:    SMOTE
  → If class weights don't give
  → enough recall, SMOTE adds
  → synthetic diversity

Avoid:     Random Oversampling
  → Overfitting risk (exact copies)
  → Only use as a baseline

Careful:   Undersampling
  → Too much data loss for our
  → 7K-row dataset
"""
axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

plt.suptitle('Class Imbalance Handling — 4 Methods Compared',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_16_imbalance_methods.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# 7.7 SAVE ALL BALANCED VERSIONS
# ============================================================================

print("\n7.7 — Saving balanced datasets")
print("─" * 60)

# Save SMOTE version (most commonly used for comparison)
X_train_smote.to_csv(os.path.join(DATA_SPLITS, 'X_train_smote.csv'), index=False)
y_train_smote.to_csv(os.path.join(DATA_SPLITS, 'y_train_smote.csv'), index=False)
print(f"  Saved: data/splits/X_train_smote.csv ({X_train_smote.shape[0]} rows)")

# Save class weights for use in model training
pd.DataFrame([weight_dict]).to_csv(os.path.join(DATA_PROCESSED, 'class_weights.csv'), index=False)
print(f"  Saved: data/processed/class_weights.csv → {weight_dict}")

# Test set is NEVER modified
print(f"\n  Test set UNCHANGED: data/splits/X_test.csv ({X_test.shape[0]} rows)")
print(f"  Test keeps real-world distribution: {y_test.mean()*100:.1f}% churn")
print()


# ============================================================================
# 7.8 WHEN TO USE WHICH METHOD — Decision Guide
# ============================================================================

print("=" * 65)
print(" 7.8 — DECISION GUIDE: When to Use Which Method")
print("=" * 65)
print("""
 Your Situation                      | Recommended Method
 ------------------------------------|------------------------------------
 First attempt, any algorithm        | Class Weights (safest default)
 Algorithm doesn't support weights   | SMOTE
 (e.g., KNN)                        |
 Need more minority diversity        | SMOTE (creates new points in gaps)
 Very large dataset (100K+ rows)     | Undersampling (saves training time)
 Mostly binary/categorical features  | Class Weights (SMOTE interpolation
                                     |   on binary features is awkward)
 Time series / sequential data       | DO NOT use SMOTE (violates temporal
                                     |   order). Use class weights.
 Extreme imbalance (99:1 or worse)   | SMOTE + Undersampling combined
                                     |   (SMOTE alone creates too many fakes)
""")


# ============================================================================
# STEP 7 SUMMARY
# ============================================================================

print("=" * 65)
print(" STEP 7 — CLASS IMBALANCE SUMMARY")
print("=" * 65)
print(f"""
 Method               | Training Size | Pros              | Cons
 ---------------------|---------------|-------------------|-------------------
 Original (no fix)    | {len(X_train):>5} rows   | Baseline          | Model ignores minority
 Random Oversampling  | {len(X_train_ros):>5} rows   | Simple            | Overfits to copies
 SMOTE                | {len(X_train_smote):>5} rows   | Synthetic variety | Binary feature artifacts
 Random Undersampling | {len(X_train_rus):>5} rows   | Fast training     | Loses majority data
 Class Weights        | {len(X_train):>5} rows   | No data change    | Not all models support

 OUR RECOMMENDATION:
   → Start with class_weight='balanced' in your model
   → If recall on churners is still low, switch to SMOTE
   → Always evaluate on the UNMODIFIED test set

 Files saved:
   • data/splits/X_train.csv / y_train.csv — original (for class weights approach)
   • data/splits/X_train_smote.csv / y_train_smote.csv — SMOTE balanced
   • data/splits/X_test.csv / y_test.csv — test set (NEVER modified)
   • data/processed/class_weights.csv — precomputed weights

 NEXT: Step 8 — Leakage Checks (final validation before modeling)
""")
print("=" * 65)
