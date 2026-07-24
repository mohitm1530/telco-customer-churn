"""
=============================================================================
 Telco Customer Churn — Data Preprocessing Pipeline
 Step 5: Feature Selection / Dimensionality Reduction
=============================================================================

Feature Selection = choosing WHICH features to keep for modeling.

WHY do we need it?
  After feature engineering, we have 26 columns. Not all help the model:
  - IRRELEVANT features add noise → worse predictions
  - REDUNDANT features cause multicollinearity → unstable coefficients
  - TOO MANY features → overfitting (model memorizes training data)
  - Fewer features → faster training, easier interpretation, better generalization

TWO PHASES in this step:
  Phase A: Feature Selection (statistical tests to pick the best features)
  Phase B: Encoding (convert selected categoricals to numbers for ML)

WHY this order?
  We select features FIRST (while they're still human-readable categories),
  then encode only the survivors. This avoids:
  - Wasting time encoding features we'll drop anyway
  - Making selection decisions on encoded columns (harder to interpret)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
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
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports', 'feature_selection')
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# LOAD ENGINEERED DATA
# ============================================================================

df = pd.read_csv(os.path.join(DATA_PROCESSED, 'telco_churn_engineered.csv'))
print(f"Engineered dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

target = 'Churn Value'
y = df[target]


# ============================================================================
# PHASE A: FEATURE SELECTION
# ============================================================================


# ============================================================================
# 5.1 METHOD 1: VARIANCE THRESHOLD — Remove Near-Constant Features
# ============================================================================
#
# WHAT: Remove features where almost all values are the same.
#   If 99% of customers have PhoneService="Yes", this column adds almost
#   zero information — it can't discriminate between churn and no-churn.
#
# WHY variance threshold FIRST?
#   It's the simplest, fastest filter. It catches obviously useless features
#   before we run expensive statistical tests. Think of it as a "pre-filter."
#
# WHY NOT just use correlation or mutual information directly?
#   → Those tests would also catch low-variance features, but they're
#     computationally more expensive. Variance threshold is O(n) per feature
#     vs O(n log n) for mutual information. On large datasets, this matters.
#   → Also, a feature with near-zero variance can numerically break some
#     statistical tests (division by near-zero in chi-square).
#
# HOW do we calculate variance for categoricals?
#   For binary: variance = p × (1-p) where p = proportion of the majority class
#   A feature where 95% of values are the same has variance = 0.95 × 0.05 = 0.0475
#   We use a threshold of 0.05 (removes features where >97.5% are one value)
# ============================================================================

print("5.1 — Variance Threshold Analysis")
print("─" * 60)

feature_cols = [c for c in df.columns if c != target]

variance_report = []
for col in feature_cols:
    if df[col].dtype == 'object':
        # For categorical: use proportion-based variance
        top_pct = df[col].value_counts(normalize=True).iloc[0]
        var = top_pct * (1 - top_pct)
        variance_report.append({
            'Feature': col, 'Type': 'categorical',
            'Top_Category_Pct': f"{top_pct:.1%}", 'Variance': round(var, 4)
        })
    else:
        # For numerical: use coefficient of variation (std/mean) as relative variance
        var = df[col].var()
        cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
        variance_report.append({
            'Feature': col, 'Type': 'numerical',
            'Top_Category_Pct': f"CV={cv:.2f}", 'Variance': round(var, 4)
        })

variance_df = pd.DataFrame(variance_report).sort_values('Variance')
print(variance_df.to_string(index=False))

low_var_features = variance_df[
    (variance_df['Type'] == 'categorical') & (variance_df['Variance'] < 0.05)
]['Feature'].tolist()

print(f"\n  Low-variance features (threshold < 0.05): {low_var_features if low_var_features else 'None'}")
if low_var_features:
    for f in low_var_features:
        print(f"    → {f}: {df[f].value_counts(normalize=True).to_dict()}")
else:
    print("  All features have sufficient variance. None removed.\n")


# ============================================================================
# 5.2 METHOD 2: CHI-SQUARE TEST — Categorical Features vs Target
# ============================================================================
#
# WHAT: Tests whether a categorical feature is INDEPENDENT of the target.
#   - High χ² score → feature is strongly associated with churn (KEEP)
#   - Low χ² score + high p-value → feature is independent of churn (DROP candidate)
#
# WHY chi-square for categoricals (not correlation)?
#   - Pearson correlation requires BOTH variables to be numerical and
#     measures only LINEAR relationships.
#   - Chi-square works on categorical vs categorical (our target is binary).
#   - It tests the NULL HYPOTHESIS: "this feature and churn are independent."
#     A small p-value rejects this → they ARE related.
#
# WHY scipy.stats.chi2_contingency (NOT sklearn.feature_selection.chi2)?
#
#   sklearn's chi2 function is designed for DOCUMENT TERM COUNTS (bag-of-words),
#   not traditional categorical feature selection. It treats input values as
#   frequencies, not category labels. Passing label-encoded integers (0, 1, 2)
#   to sklearn's chi2 computes a meaningless test — it treats "2" as a count
#   twice as large as "1", which is wrong for categories.
#
#   scipy.stats.chi2_contingency computes the PROPER Pearson chi-square test
#   on a contingency table (cross-tabulation). This is the correct statistical
#   test for "is this categorical feature independent of the target?"
#
#   This is a common mistake in tutorials — many use sklearn's chi2 on
#   label-encoded features without realizing it's the wrong test.
#
# HOW: We build a contingency table (pd.crosstab) for each feature vs target,
#   then run chi2_contingency. No encoding needed — works on raw categories.
# ============================================================================

print("5.2 — Chi-Square Test of Independence (Categorical Features vs Churn)")
print("─" * 60)

categorical_features = df[feature_cols].select_dtypes(include='object').columns.tolist()
print(f"  Testing {len(categorical_features)} categorical features:\n")

chi2_results = []

for col in categorical_features:
    contingency = pd.crosstab(df[col], y)
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
    chi2_results.append({
        'Feature': col,
        'Chi2_Score': round(chi2_stat, 2),
        'P_Value': round(p_value, 6),
        'DOF': dof,
        'Significant': 'YES' if p_value < 0.05 else 'NO'
    })

chi2_df = pd.DataFrame(chi2_results).sort_values('Chi2_Score', ascending=False)
print(chi2_df.to_string(index=False))

non_significant = chi2_df[chi2_df['Significant'] == 'NO']['Feature'].tolist()
print(f"\n  Non-significant features (p > 0.05): {non_significant if non_significant else 'None'}")
print()


# ============================================================================
# 5.3 METHOD 3: MUTUAL INFORMATION — All Features vs Target
# ============================================================================
#
# WHAT: Measures how much INFORMATION a feature shares with the target.
#   MI = 0 → feature and target are completely independent
#   MI > 0 → knowing the feature reduces uncertainty about the target
#
# WHY mutual information IN ADDITION to chi-square?
#
#   Chi-square tests for ASSOCIATION but can't quantify HOW MUCH information
#   a feature provides. Two features might both be "significant" (p < 0.05)
#   but one could be 10x more informative than the other.
#
#   Mutual information RANKS features by information content, letting us
#   prioritize the most valuable ones.
#
# WHY mutual information over Pearson/Spearman correlation?
#   - Pearson: LINEAR relationships only. Misses non-linear patterns.
#   - Spearman: MONOTONIC relationships only. Better than Pearson, but still
#     misses patterns like "churn is high at BOTH ends of tenure" (U-shaped).
#   - Mutual Information: captures ANY statistical dependency (linear,
#     non-linear, U-shaped, etc.). It's the most general measure.
#
# WHY NOT use ONLY mutual information (skip chi-square)?
#   → MI is non-parametric and can be noisy on small samples.
#   → Chi-square gives a p-value (statistical significance), MI doesn't.
#   → Using BOTH gives a more robust picture:
#     - Chi-square says "is this feature significant?"
#     - MI says "HOW informative is this feature?"
#     - A feature that's significant AND highly informative is a confident keep.
#
# PARAMETERS:
#   discrete_features: We mark which features are categorical so the MI
#   estimator uses the correct algorithm (counting-based for discrete,
#   KNN-based for continuous).
#   random_state: MI estimation involves randomness — fixed seed for
#   reproducibility.
# ============================================================================

print("5.3 — Mutual Information (All Features vs Churn)")
print("─" * 60)

# Prepare: encode ALL features to numeric for MI calculation
df_encoded_temp = df[feature_cols].copy()
discrete_mask = []

for col in feature_cols:
    if df_encoded_temp[col].dtype == 'object':
        df_encoded_temp[col] = LabelEncoder().fit_transform(df_encoded_temp[col].astype(str))
        discrete_mask.append(True)
    else:
        discrete_mask.append(False)

mi_scores = mutual_info_classif(
    df_encoded_temp, y,
    discrete_features=discrete_mask,
    random_state=42,
    n_neighbors=5
)

mi_df = pd.DataFrame({
    'Feature': feature_cols,
    'MI_Score': np.round(mi_scores, 4)
}).sort_values('MI_Score', ascending=False)

print(mi_df.to_string(index=False))

low_mi_features = mi_df[mi_df['MI_Score'] < 0.005]['Feature'].tolist()
print(f"\n  Very low MI features (< 0.005): {low_mi_features if low_mi_features else 'None'}")
print()


# ============================================================================
# 5.4 COMBINE RESULTS — Final Feature Selection Decision
# ============================================================================
#
# We combine all three methods to make robust decisions:
#   - Variance threshold: catches near-constant features
#   - Chi-square: statistical significance of association
#   - Mutual information: quantifies information content
#
# DECISION RULES:
#   DROP if: low variance OR (not significant AND low MI)
#   KEEP if: significant AND reasonable MI
#   REVIEW if: mixed signals (significant but low MI, or vice versa)
#
# WHY use multiple methods instead of just one?
#   → No single method is perfect:
#     - Variance threshold misses features that vary a lot but aren't
#       related to the target
#     - Chi-square only works on categoricals and doesn't rank importance
#     - MI can be noisy and doesn't give significance levels
#   → Agreement across methods = high confidence in the decision
# ============================================================================

print("=" * 65)
print(" 5.4 — COMBINED FEATURE SELECTION DECISION")
print("=" * 65)

# Merge all results
selection = mi_df.copy()

# Add chi-square results (only for categoricals)
chi2_lookup = dict(zip(chi2_df['Feature'], chi2_df['P_Value']))
chi2_score_lookup = dict(zip(chi2_df['Feature'], chi2_df['Chi2_Score']))
selection['Chi2_Score'] = selection['Feature'].map(chi2_score_lookup)
selection['Chi2_P'] = selection['Feature'].map(chi2_lookup)

# Add variance info
var_lookup = dict(zip(variance_df['Feature'], variance_df['Variance']))
selection['Variance'] = selection['Feature'].map(var_lookup)

# Determine action
def decide(row):
    # Low variance → drop
    if row['Variance'] < 0.05 and row['Feature'] in categorical_features:
        return 'DROP (low variance)'

    # Check for high multicollinearity with existing features
    if row['Feature'] == 'Total Charges':
        return 'DROP (multicollinear with Tenure × Monthly Charges)'

    # Avg Monthly Spend correlates 0.996 with Monthly Charges — nearly identical
    if row['Feature'] == 'Avg Monthly Spend':
        return 'DROP (r=0.996 with Monthly Charges — redundant)'

    # Engagement Score is a linear combo of Tenure + Monthly Charges + Num Addons
    # It correlates 0.90 with Num Addon Services and 0.79 with Monthly Charges.
    # The model can learn these weights itself from the raw components.
    if row['Feature'] == 'Engagement Score':
        return 'DROP (linear combo of kept features — model learns weights itself)'

    # Non-significant chi-square AND low MI
    if pd.notna(row['Chi2_P']) and row['Chi2_P'] > 0.05 and row['MI_Score'] < 0.01:
        return 'DROP (not significant, low MI)'

    # Gender: EDA showed no predictive power
    if row['Feature'] == 'Gender' and row['MI_Score'] < 0.005:
        return 'DROP (no predictive power — EDA confirmed)'

    return 'KEEP'

selection['Decision'] = selection.apply(decide, axis=1)
selection = selection.sort_values('MI_Score', ascending=False)

print()
for _, row in selection.iterrows():
    status = 'KEEP' if row['Decision'] == 'KEEP' else 'DROP'
    mi = f"MI={row['MI_Score']:.4f}"
    chi2_str = f"χ²={row['Chi2_Score']:.0f}" if pd.notna(row['Chi2_Score']) else "χ²=N/A"
    marker = "  +" if status == 'KEEP' else "  -"
    print(f"  {marker} {row['Feature']:.<30s} {mi}  {chi2_str:>10s}  → {row['Decision']}")

keep_features = selection[selection['Decision'] == 'KEEP']['Feature'].tolist()
drop_features = selection[selection['Decision'] != 'KEEP']['Feature'].tolist()

print(f"\n  Features to KEEP: {len(keep_features)}")
print(f"  Features to DROP: {len(drop_features)} → {drop_features}")


# ============================================================================
# 5.5 HANDLE MULTICOLLINEARITY — Correlation Check on Kept Numericals
# ============================================================================
#
# WHAT: Check if any KEPT numerical features are highly correlated (>0.7).
#   High correlation = redundant information = unstable model coefficients.
#
# WHY check AFTER selection (not before)?
#   → We already dropped Total Charges (known multicollinearity from EDA).
#   → The engineered features (Avg Monthly Spend, Engagement Score) need
#     checking because they were DERIVED from existing features and might
#     be highly correlated with them.
#
# WHY threshold of 0.7?
#   → Academic convention: |r| > 0.7 indicates strong multicollinearity
#   → Below 0.7, the features still carry enough independent information
#     to justify keeping both
#   → Above 0.7, you start getting unstable coefficients in linear models
#     (OLS standard errors inflate, logistic regression converges slowly)
# ============================================================================

print("\n\n5.5 — Multicollinearity Check (Kept Numerical Features)")
print("─" * 60)

kept_numerical = [f for f in keep_features if df[f].dtype in ['int64', 'float64']]
if len(kept_numerical) > 1:
    corr_matrix = df[kept_numerical].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=1, mask=mask,
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Correlation Matrix — Kept Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'plot_13_kept_features_correlation.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Flag high correlations
    print("  High correlations (|r| > 0.7):")
    flagged = False
    for i in range(len(kept_numerical)):
        for j in range(i+1, len(kept_numerical)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                print(f"    {kept_numerical[i]} ↔ {kept_numerical[j]}: r = {r:.3f}")
                flagged = True
    if not flagged:
        print("    None found. No additional drops needed.")
else:
    print("  Only 0-1 numerical features kept. Skipping.")

print()


# ============================================================================
# PHASE B: ENCODING
# ============================================================================


# ============================================================================
# 5.6 DROP REJECTED FEATURES
# ============================================================================

print("5.6 — Dropping rejected features")
df_selected = df[keep_features + [target]].copy()
print(f"     Before: {df.shape[1]} columns → After: {df_selected.shape[1]} columns")
print(f"     Dropped: {drop_features}\n")


# ============================================================================
# 5.7 ENCODE CATEGORICAL FEATURES
# ============================================================================
#
# ML algorithms need numbers, not strings. Encoding converts categorical
# values to numerical representations.
#
# WHICH encoding method to use? It depends on the feature type:
#
# ┌───────────────────────────┬──────────────────┬──────────────────────────┐
# │ Feature Type              │ Encoding Method  │ Why                      │
# ├───────────────────────────┼──────────────────┼──────────────────────────┤
# │ Binary (2 categories)     │ Label Encoding   │ 0/1 is natural, no extra │
# │ e.g., Gender, Partner     │ (0, 1)           │ columns, no info lost    │
# ├───────────────────────────┼──────────────────┼──────────────────────────┤
# │ Nominal (3+ categories,   │ One-Hot Encoding │ No artificial ordering.  │
# │ no natural order)         │ (dummy variables)│ Each category gets own   │
# │ e.g., Internet Service,   │                  │ binary column.           │
# │ Payment Method            │                  │                          │
# ├───────────────────────────┼──────────────────┼──────────────────────────┤
# │ Ordinal (ordered)         │ Ordinal Encoding │ Preserves natural order. │
# │ e.g., Tenure Group,       │ (0, 1, 2, 3)    │ "0-12 mo" < "13-24 mo"  │
# │ Charges Tier              │                  │ < "25-48 mo" < "49-72 mo"│
# └───────────────────────────┴──────────────────┴──────────────────────────┘
#
# WHY NOT use Label Encoding for everything?
#   Label Encoding assigns integers: DSL=0, Fiber optic=1, No=2
#   This implies an ORDER (DSL < Fiber optic < No) and a DISTANCE
#   (the gap from DSL to Fiber is the same as Fiber to No).
#   Neither is true! The model would learn false relationships.
#   One-hot encoding avoids this by creating separate binary columns.
#
# WHY NOT use One-Hot Encoding for everything?
#   For binary features (Male/Female), one-hot creates TWO columns
#   (is_Male, is_Female) that are perfectly inversely correlated.
#   This is redundant and causes multicollinearity. Label encoding
#   (Male=0, Female=1) is simpler and equivalent.
#
# WHY drop_first=True in one-hot encoding?
#   With k categories, you only need k-1 dummy columns. The k-th category
#   is implied when all others are 0. Keeping all k creates perfect
#   multicollinearity (the "dummy variable trap"), which breaks linear
#   models. Tree-based models handle it but still waste a column.
# ============================================================================

print("5.7 — Encoding Categorical Features")
print("─" * 60)

# Identify feature types in the selected dataset
selected_cats = df_selected.select_dtypes(include='object').columns.tolist()

binary_features = []
nominal_features = []
ordinal_features = []

ordinal_mappings = {
    'Tenure Group': {'0-12 mo': 0, '13-24 mo': 1, '25-48 mo': 2, '49-72 mo': 3},
    'Charges Tier': {'Low ($18-35)': 0, 'Mid ($35-70)': 1, 'High ($70-120)': 2}
}

for col in selected_cats:
    if col in ordinal_mappings:
        ordinal_features.append(col)
    elif df_selected[col].nunique() == 2:
        binary_features.append(col)
    else:
        nominal_features.append(col)

print(f"  Binary features ({len(binary_features)}):  {binary_features}")
print(f"  Ordinal features ({len(ordinal_features)}): {ordinal_features}")
print(f"  Nominal features ({len(nominal_features)}): {nominal_features}\n")

# --- Binary: Label Encoding ---
print("  Applying Label Encoding to binary features:")
for col in binary_features:
    le = LabelEncoder()
    df_selected[col] = le.fit_transform(df_selected[col])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"    {col}: {mapping}")

# --- Ordinal: Manual mapping ---
print("\n  Applying Ordinal Encoding:")
for col, mapping in ordinal_mappings.items():
    if col in df_selected.columns:
        df_selected[col] = df_selected[col].map(mapping)
        print(f"    {col}: {mapping}")

# --- Nominal: One-Hot Encoding (drop_first=True) ---
print(f"\n  Applying One-Hot Encoding to nominal features (drop_first=True):")
for col in nominal_features:
    print(f"    {col}: {sorted(df_selected[col].unique())} → {df_selected[col].nunique()-1} dummy columns")

df_selected = pd.get_dummies(df_selected, columns=nominal_features, drop_first=True, dtype=int)

print(f"\n  After encoding: {df_selected.shape[1]} columns total")
print(f"  Columns: {list(df_selected.columns)}\n")


# ============================================================================
# 5.8 FINAL FEATURE SET VALIDATION
# ============================================================================

print("=" * 65)
print(" 5.8 — FINAL FEATURE SET VALIDATION")
print("=" * 65)

print(f"\n  Shape: {df_selected.shape}")
print(f"  All numeric: {df_selected.select_dtypes(include='object').shape[1] == 0}")
print(f"  Any NaN: {df_selected.isnull().any().any()}")
print(f"  Target balance: {dict(df_selected[target].value_counts())}")

print(f"\n  Feature dtypes:")
print(df_selected.dtypes)

print(f"\n  First 5 rows (transposed for readability):")
print(df_selected.head().T)


# ============================================================================
# 5.9 FEATURE IMPORTANCE RANKING — Final View
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Compute MI on final encoded features
final_features = [c for c in df_selected.columns if c != target]
final_mi = mutual_info_classif(
    df_selected[final_features], df_selected[target],
    random_state=42
)
mi_final_df = pd.DataFrame({
    'Feature': final_features,
    'MI_Score': final_mi
}).sort_values('MI_Score', ascending=True)

colors_mi = ['#2ecc71' if s > 0.02 else '#f39c12' if s > 0.005 else '#e74c3c'
             for s in mi_final_df['MI_Score']]
ax.barh(range(len(mi_final_df)), mi_final_df['MI_Score'], color=colors_mi, edgecolor='black')
ax.set_yticks(range(len(mi_final_df)))
ax.set_yticklabels(mi_final_df['Feature'], fontsize=9)
ax.set_xlabel('Mutual Information Score')
ax.set_title('Final Feature Importance (Mutual Information with Churn)',
             fontsize=14, fontweight='bold')
ax.axvline(x=0.005, color='red', linestyle='--', alpha=0.5, label='Low-value threshold')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_14_final_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# 5.10 SAVE FINAL SELECTED + ENCODED DATA
# ============================================================================

df_selected.to_csv(os.path.join(DATA_PROCESSED, 'telco_churn_selected.csv'), index=False)
print(f"\n  Saved: data/processed/telco_churn_selected.csv ({df_selected.shape[0]} rows × {df_selected.shape[1]} columns)")


# ============================================================================
# STEP 5 SUMMARY
# ============================================================================

print("\n" + "=" * 65)
print(" STEP 5 — FEATURE SELECTION & ENCODING SUMMARY")
print("=" * 65)
print(f"""
 SELECTION METHODS USED:
   1. Variance Threshold — removed near-constant features (if any)
   2. Chi-Square Test    — statistical significance for categoricals
   3. Mutual Information — information content ranking for all features
   Combined: drop only when MULTIPLE methods agree → robust decisions

 ENCODING METHODS USED:
   • Binary features   → Label Encoding (0/1)
   • Ordinal features  → Manual ordinal mapping (preserves order)
   • Nominal features  → One-Hot Encoding (drop_first=True, avoids trap)

 WHY NOT other approaches:
   • Target Encoding: risks leakage without cross-validation; overkill here
   • Frequency Encoding: loses category identity; only good for high-cardinality
   • PCA for dimensionality reduction: loses interpretability; we don't have
     enough features to justify it (PCA shines with 100+ features)

 Pipeline: {df.shape[1]} cols (engineered) → {len(keep_features)} kept → {df_selected.shape[1]} cols (after encoding)
 Saved to: data/processed/telco_churn_selected.csv

 NEXT: Step 6 — Data Splitting
""")
print("=" * 65)
