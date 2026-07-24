"""
=============================================================================
 Telco Customer Churn — Data Preprocessing Pipeline
 Step 3: Data Cleaning
=============================================================================

Data Cleaning = fixing the raw data so it's ready for analysis and modeling.

This step addresses everything we flagged in Steps 1-2:
  1. Remove the blank row
  2. Drop useless / leakage columns
  3. Fix Total Charges (string → numeric)
  4. Handle missing values (the 11 space-only Total Charges rows)
  5. Standardize categorical values
  6. Fix data types
  7. Validate the cleaned dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports', 'cleaning')
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# LOAD RAW DATA
# ============================================================================

df = pd.read_csv(os.path.join(DATA_RAW, 'telco_churn.csv'))
print(f"Raw dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")


# ============================================================================
# 3.1 REMOVE THE COMPLETELY BLANK ROW
# ============================================================================
#
# What we found: Row 7043 has NaN in EVERY column (except Churn Reason which
# is NaN for most rows anyway). This is not missing data — it's an empty record
# that shouldn't exist.
#
# Why DROP (not impute)?
#   - Imputation makes sense when you have PARTIAL information about a record
#     (e.g., a customer with known gender but unknown tenure). You can estimate
#     the missing values from what you DO know.
#   - This row has ZERO information — there's nothing to estimate FROM.
#   - Losing 1 row out of 7,044 (0.01%) has zero impact on model performance.
#
# Why not just dropna() on all columns?
#   - dropna() would also remove the 11 rows where only Total Charges is
#     missing (as a space). We DON'T want to lose those — they're real customers
#     with tenure=0 (brand new). We'll handle them separately.
# ============================================================================

cols_for_null_check = [c for c in df.columns if c != 'Churn Reason']
blank_mask = df[cols_for_null_check].isnull().all(axis=1)
print(f"3.1 — Completely blank rows found: {blank_mask.sum()}")
df = df[~blank_mask].reset_index(drop=True)
print(f"     After removal: {df.shape[0]} rows\n")


# ============================================================================
# 3.2 DROP USELESS AND LEAKAGE COLUMNS
# ============================================================================
#
# Three categories of columns to remove, each for a different reason:
#
# CATEGORY 1: Zero-Variance / Constant Columns
# ─────────────────────────────────────────────
#   Column   | Values          | Why Drop
#   ---------|-----------------|--------------------------------------------
#   Count    | All 1           | A constant adds zero information. Variance
#            |                 | = 0, so no algorithm can learn from it.
#   Country  | All "United     | Same reason. Every customer is in the US,
#            | States"         | so this column has no discriminating power.
#   State    | All "California"| Same. All customers are in CA.
#
#   WHY NOT KEEP THEM "just in case"?
#   → They can't help prediction (zero variance = zero correlation with
#     anything). They only waste memory and add noise.
#
#
# CATEGORY 2: Redundant / Identifier Columns
# ───────────────────────────────────────────
#   Column     | Why Drop
#   -----------|--------------------------------------------------------
#   CustomerID | Unique per row. If kept, tree-based models might memorize
#              | IDs instead of learning patterns. Linear models ignore it
#              | (no correlation), but it still wastes space.
#   Lat Long   | String like "33.96, -118.27". We already have Latitude
#              | and Longitude as separate numeric columns. Keeping both
#              | is redundant, and the string version can't be used
#              | directly by any algorithm.
#   Churn Label| Text version of Churn Value ("Yes"/"No" vs 1/0). We use
#              | Churn Value as our target — keeping both would mean the
#              | model could "cheat" by looking at the label.
#
#
# CATEGORY 3: Data Leakage Columns
# ─────────────────────────────────
#   Column       | Why It's Leakage
#   -------------|--------------------------------------------------------
#   Churn Reason | ONLY populated for customers who already churned.
#                | At prediction time (for a current customer), you wouldn't
#                | know their churn reason — because they haven't churned yet.
#                | Using it = using future information → artificially perfect
#                | model that fails on real data.
#   Churn Score  | A pre-computed churn probability (corr=0.65 with target).
#                | If this was computed using Churn Value, it's circular
#                | reasoning. Even if not, using a "pre-made prediction" as
#                | a feature defeats the purpose of building our own model.
#   CLTV         | Customer Lifetime Value. Often calculated using whether
#                | the customer stayed or left (churn status). If churn was
#                | used in CLTV calculation, it's target leakage.
#
#   WHAT IS DATA LEAKAGE?
#   → When your training data contains information that wouldn't be available
#     at prediction time. The model learns to rely on this "cheat" information,
#     gets amazing test scores, but fails catastrophically in production.
#   → Leakage is the #1 reason ML projects succeed in the lab and fail in
#     the real world.
#
#
# CATEGORY 4: High-Cardinality Geographic Columns
# ────────────────────────────────────────────────
#   Column    | Unique Values | Why Drop
#   ----------|---------------|----------------------------------------------
#   City      | 1,129         | Too many categories. Encoding 1,129 cities
#             |               | creates 1,128 dummy variables, most with <10
#             |               | samples → model overfits to noise.
#   Zip Code  | ~1,600        | Same problem, even worse cardinality.
#   Latitude  | continuous    | Could theoretically help (cluster by region),
#   Longitude | continuous    | but EDA showed geographic variation is small
#             |               | (22-32% churn across top cities). The signal
#             |               | doesn't justify the complexity.
#
#   WHY NOT reduce cardinality (e.g., group into regions)?
#   → We could, but EDA showed the churn rate variation across cities is
#     small (~10 percentage points) compared to Contract type (~40 pp).
#     The effort-to-signal ratio is poor. If this were a location-heavy
#     business (e.g., retail), we'd invest more here.
# ============================================================================

drop_cols = [
    # Category 1: Constants
    'Count', 'Country', 'State',
    # Category 2: Redundant / Identifier
    'CustomerID', 'Lat Long', 'Churn Label',
    # Category 3: Leakage
    'Churn Reason', 'Churn Score', 'CLTV',
    # Category 4: High-cardinality geographic
    'City', 'Zip Code', 'Latitude', 'Longitude'
]

print(f"3.2 — Dropping {len(drop_cols)} columns:")
for col in drop_cols:
    print(f"     • {col}")

df = df.drop(columns=drop_cols)
print(f"\n     Remaining: {df.shape[1]} columns")
print(f"     Columns: {list(df.columns)}\n")


# ============================================================================
# 3.3 FIX TOTAL CHARGES — STRING TO NUMERIC
# ============================================================================
#
# Problem: Total Charges is stored as `object` (string) instead of `float`.
#
# Root cause: 11 rows contain a single space ' ' instead of a number.
# These are all customers with Tenure Months = 0 (brand new sign-ups who
# haven't been billed yet).
#
# Step 1: Convert to numeric using pd.to_numeric(errors='coerce')
# ──────────────────────────────────────────────────────────────
#
#   Why `errors='coerce'` and not `errors='raise'`?
#   → 'raise' would throw an exception and stop execution at the first
#     non-numeric value. We'd have to manually clean BEFORE converting.
#   → 'coerce' converts non-numeric to NaN, letting us handle them after.
#   → This is the standard approach because it's defensive — it works even
#     if there are unexpected non-numeric values beyond the spaces.
#
#   Why not .str.strip() + .astype(float)?
#   → .str.strip() only handles whitespace. If there were other non-numeric
#     values (like '$108.15' or 'N/A'), it would still fail.
#   → pd.to_numeric(errors='coerce') handles ALL non-numeric cases.
#
#
# Step 2: Fill the resulting NaN values
# ─────────────────────────────────────
#   The 11 NaN rows are tenure=0 customers. Their Total Charges should
#   logically be 0 (never billed). We fill with 0.
#
#   Why fill with 0 (not mean, median, or drop)?
#   → DOMAIN KNOWLEDGE: Total Charges = sum of all bills. If tenure=0
#     (no billing cycles), total charges = 0. This is the only correct value.
#   → Mean imputation would assign ~$2,280 to customers who were never
#     billed — factually wrong and would distort the model.
#   → Median imputation (~$1,397) — same problem, wrong value.
#   → Dropping these 11 rows loses real customers. They're brand new
#     sign-ups who might churn early — exactly the pattern we want to detect.
#   → KNN/regression imputation would predict based on other features,
#     but we KNOW the answer: it's 0. Using a statistical estimate when
#     you have domain knowledge is worse, not better.
#
#   GENERAL RULE: Always prefer domain knowledge over statistical imputation
#   when you KNOW what the correct value is. Statistics is for when you DON'T.
# ============================================================================

print("3.3 — Fixing Total Charges")
print(f"     Before: dtype = {df['Total Charges'].dtype}")

# Show the problem rows
problem_mask = pd.to_numeric(df['Total Charges'], errors='coerce').isna()
print(f"     Non-numeric values found: {problem_mask.sum()}")
print(f"     All have Tenure=0: {(df.loc[problem_mask, 'Tenure Months'] == 0).all()}")

# Convert to numeric (spaces → NaN)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Fill NaN with 0 (tenure=0 → never billed → total charges = 0)
df['Total Charges'] = df['Total Charges'].fillna(0)

print(f"     After: dtype = {df['Total Charges'].dtype}")
print(f"     Remaining NaNs: {df['Total Charges'].isna().sum()}\n")


# ============================================================================
# 3.4 HANDLE "NO PHONE SERVICE" / "NO INTERNET SERVICE" VALUES
# ============================================================================
#
# Several columns have three values: "Yes", "No", "No phone/internet service"
#
#   Multiple Lines:   Yes / No / "No phone service"
#   Online Security:  Yes / No / "No internet service"
#   Online Backup:    Yes / No / "No internet service"
#   Device Protection: Yes / No / "No internet service"
#   Tech Support:     Yes / No / "No internet service"
#   Streaming TV:     Yes / No / "No internet service"
#   Streaming Movies: Yes / No / "No internet service"
#
# DECISION: Keep them as-is. Here's why:
#
#   Option A: Replace "No internet service" with "No"
#   → LOSES information. "No online security" (has internet but chose not to
#     add security) is DIFFERENT from "No internet service" (can't have
#     security because no internet). EDA showed these groups have different
#     churn rates: "No internet service" = ~7% churn, "No" = ~42% churn.
#     Collapsing them destroys a powerful signal.
#
#   Option B: Keep as 3 categories (our choice)
#   → Preserves the distinction. When we one-hot encode later, each gets
#     its own binary column, and the model can learn different coefficients
#     for each.
#
#   Option C: Create a separate binary "has_internet" feature and then
#     replace "No internet service" with "No"
#   → This is valid too, but adds complexity. The 3-category approach
#     captures the same information more simply. The model can learn
#     the "has internet" signal from the "No internet service" category.
# ============================================================================

print("3.4 — Categorical value audit")
service_cols = ['Multiple Lines', 'Online Security', 'Online Backup',
                'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']

for col in service_cols:
    vals = df[col].value_counts()
    print(f"     {col}: {dict(vals)}")

print("\n     DECISION: Keeping 'No phone/internet service' as distinct categories.")
print("     Reason: They have different churn rates than plain 'No' (~7% vs ~42%).\n")


# ============================================================================
# 3.5 VERIFY DATA TYPES
# ============================================================================
#
# Every column should have the correct dtype for what it represents:
#   - Numerical features → float64 or int64
#   - Categorical features → object (string) or category
#   - Target → int (0/1)
#
# Why does dtype matter?
#   - Wrong dtype = wrong processing. If a categorical column is stored as
#     int (e.g., Senior Citizen as 0/1), scaling or normalization would treat
#     it as continuous — applying StandardScaler to a binary variable is
#     meaningless and can confuse some algorithms.
#   - Pandas operations differ by dtype: .value_counts() for categoricals,
#     .mean()/.std() for numericals.
#   - Memory efficiency: 'category' dtype uses less memory than 'object'.
#
# In our case, all dtypes are already correct after the Total Charges fix:
#   - Numerical: Tenure Months (float), Monthly Charges (float), Total Charges (float)
#   - Categorical: all others (object / string)
#   - Target: Churn Value (float → we'll convert to int)
# ============================================================================

print("3.5 — Data type verification")
print(df.dtypes)

# Convert target to int (it's currently float due to the original NaN row)
df['Churn Value'] = df['Churn Value'].astype(int)
print(f"\n     Churn Value converted: float → {df['Churn Value'].dtype}")

# Convert Tenure Months to int (no fractional months)
df['Tenure Months'] = df['Tenure Months'].astype(int)
print(f"     Tenure Months converted: float → {df['Tenure Months'].dtype}\n")


# ============================================================================
# 3.6 CHECK FOR DUPLICATES
# ============================================================================
#
# Why check for duplicates?
#   - Duplicate rows inflate the importance of that data point.
#   - If a churned customer appears 5 times, the model over-weights that
#     customer's profile, leading to bias.
#   - In train/test split, duplicates could appear in both sets → data leakage
#     (model has "seen" the test data during training).
#
# IMPORTANT DISTINCTION: True duplicates vs coincidental matches
#   - A TRUE duplicate = the same real-world entity recorded twice
#     (e.g., same customer entered into the system twice by mistake).
#   - A COINCIDENTAL MATCH = two different customers who happen to have
#     identical demographics and service plans. This is NOT a duplicate —
#     it's two real people who look alike on paper.
#
# How to tell them apart?
#   - The ONLY reliable way is to check BEFORE dropping the unique identifier
#     (CustomerID). If two rows have DIFFERENT CustomerIDs but identical
#     features, they are different customers — keep both.
#   - If we check for duplicates AFTER dropping CustomerID (as we did
#     initially — a bug!), we'd falsely remove 22 real customers who just
#     happen to share the same plan and demographics.
#
# LESSON: Always check duplicates on the FULL dataset (with IDs) BEFORE
#   dropping identifier columns, or explicitly exclude the ID check.
# ============================================================================

# We already dropped CustomerID, so we can't check by ID anymore.
# But we verified in our raw data exploration that:
#   - 0 duplicate CustomerIDs existed → every row was a unique customer
#   - So any "duplicates" after dropping CustomerID are coincidental matches
#     between different customers, NOT true duplicates.

n_feature_matches = df.duplicated().sum()
print(f"3.6 — Rows with identical feature values: {n_feature_matches}")
print("     These are NOT true duplicates — they are different customers")
print("     who happen to share the same demographics and plan.")
print("     We verified 0 duplicate CustomerIDs in the raw data.")
print("     DECISION: Keep all rows. Removing would lose real customers.\n")


# ============================================================================
# 3.7 OUTLIER ANALYSIS — SHOULD WE REMOVE OR CAP?
# ============================================================================
#
# EDA (Step 2) showed no extreme outliers in our numerical features.
# But let's do a formal check using the IQR method and discuss WHY we're
# choosing NOT to remove them.
#
# WHAT IS THE IQR METHOD?
#   IQR = Q3 - Q1 (interquartile range = middle 50% of data)
#   Lower bound = Q1 - 1.5 × IQR
#   Upper bound = Q3 + 1.5 × IQR
#   Any value outside these bounds is an "outlier" by this definition.
#
# WHY IQR (not Z-score)?
#   - Z-score assumes NORMAL distribution. If data is skewed (like Total
#     Charges), Z-score misidentifies normal tail values as outliers.
#   - IQR is distribution-free — works regardless of shape.
#   - IQR is also robust to the outliers themselves (mean and std used in
#     Z-score are pulled by outliers, but Q1/Q3 are not).
#
# WHY WE'RE NOT REMOVING OUTLIERS HERE:
#   1. Monthly Charges: ranges $18-$119. This is the real billing range —
#      there's no "impossible" value. Removing high payers would bias the
#      model against premium plan customers.
#   2. Tenure: 0-72 months (0-6 years). All plausible.
#   3. Total Charges: right-skewed, but that's expected (tenure × monthly).
#      The "outliers" are just long-term high-paying customers — real and
#      important data points.
#
#   GENERAL RULE for outlier handling:
#   → Remove: if the value is clearly an ERROR (e.g., age=999, salary=-50000)
#   → Cap (winsorize): if extreme values exist but are real, and you want to
#     reduce their influence without losing the data point
#   → Keep: if the values are within plausible range and represent real
#     variation in the population (OUR CASE)
# ============================================================================

print("3.7 — Outlier analysis (IQR method)")
numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']

for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"     {col}:")
    print(f"       Range: [{df[col].min():.1f}, {df[col].max():.1f}]")
    print(f"       IQR bounds: [{lower:.1f}, {upper:.1f}]")
    print(f"       Outliers: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

print("\n     DECISION: No outlier removal. All values are within plausible")
print("     business ranges. 'Outliers' are just high-tenure or high-bill")
print("     customers — real and important data points.\n")


# ============================================================================
# 3.8 FINAL VALIDATION
# ============================================================================
#
# Before saving, verify the cleaned dataset is in good shape:
#   1. No remaining NaN values (in features we'll use)
#   2. Correct data types
#   3. Expected number of rows/columns
#   4. Target variable is clean
# ============================================================================

print("=" * 65)
print(" 3.8 — FINAL VALIDATION")
print("=" * 65)

print(f"\n  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print(f"\n  Missing values per column:")
nulls = df.isnull().sum()
print(f"  {dict(nulls[nulls > 0]) if nulls.sum() > 0 else '  None — all clean!'}")

print(f"\n  Data types:")
for dtype_name, count in df.dtypes.value_counts().items():
    cols = df.select_dtypes(include=[dtype_name]).columns.tolist()
    print(f"    {dtype_name}: {count} columns → {cols}")

print(f"\n  Target distribution:")
print(f"    {dict(df['Churn Value'].value_counts())}")
print(f"    Churn rate: {df['Churn Value'].mean()*100:.1f}%")

print(f"\n  Numerical feature stats:")
print(df[numerical_cols].describe().to_string())

print(f"\n  Categorical feature value counts:")
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"    {col}: {df[col].nunique()} categories → {sorted(df[col].unique())}")


# ============================================================================
# 3.9 SAVE CLEANED DATA
# ============================================================================

df.to_csv(os.path.join(DATA_PROCESSED, 'telco_churn_cleaned.csv'), index=False)
print(f"\n  Saved: data/processed/telco_churn_cleaned.csv ({df.shape[0]} rows × {df.shape[1]} columns)")


# ============================================================================
# 3.10 WHAT CLEANING ACTUALLY ACCOMPLISHED — VISUAL SUMMARY
# ============================================================================
#
# Why NOT a "before vs after" histogram?
#   The numerical distributions barely changed — and that's EXPECTED.
#   Data cleaning in this dataset was STRUCTURAL, not distributional:
#     - We removed dangerous leakage columns (prevents fake accuracy)
#     - We fixed data types (prevents crashes and wrong calculations)
#     - We filled 11 logically-correct zeros (prevents losing real customers)
#   None of these change the shape of a histogram. Showing near-identical
#   histograms side-by-side would be misleading — it would suggest cleaning
#   "didn't do anything", when in reality it fixed critical issues that
#   would silently break modeling.
#
# Instead, we show WHAT was actually fixed and WHY it matters.
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Plot 1: Column reduction (what we kept vs dropped) ---
raw_df = pd.read_csv(os.path.join(DATA_RAW, 'telco_churn.csv'))
kept_cols = len(df.columns)
dropped_cols = len(raw_df.columns) - kept_cols
categories = ['Kept\n(useful features)', 'Dropped\n(constants/IDs)', 'Dropped\n(leakage)', 'Dropped\n(high-cardinality)']
values = [kept_cols, 6, 3, 4]
colors_bar = ['#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c']

axes[0, 0].bar(categories, values, color=colors_bar, edgecolor='black')
for i, v in enumerate(values):
    axes[0, 0].text(i, v + 0.3, str(v), ha='center', fontweight='bold', fontsize=13)
axes[0, 0].set_title('Column Audit: 33 → 20 columns', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Number of Columns')

# --- Plot 2: Total Charges fix — before vs after for the 11 problem rows ---
raw_tc = pd.to_numeric(raw_df['Total Charges'], errors='coerce')
problem_idx = raw_tc.isna() & raw_df['Total Charges'].notna()
problem_rows = raw_df[problem_idx][['Tenure Months', 'Monthly Charges']].copy()
problem_rows['Total Charges (Before)'] = 'NaN (space)'
problem_rows['Total Charges (After)'] = 0.0

axes[0, 1].barh(range(len(problem_rows)), [0]*len(problem_rows), color='#e74c3c', height=0.4, label='Before: " " (space)')
axes[0, 1].barh([x+0.4 for x in range(len(problem_rows))], [0]*len(problem_rows), color='#2ecc71', height=0.4, label='After: 0.0')
axes[0, 1].set_yticks([x+0.2 for x in range(len(problem_rows))])
axes[0, 1].set_yticklabels([f'Tenure=0, ${mc:.0f}/mo' for mc in problem_rows['Monthly Charges']], fontsize=9)
axes[0, 1].set_title(f'Total Charges Fix: {len(problem_rows)} rows with " " → 0', fontsize=13, fontweight='bold')
axes[0, 1].text(0.5, 0.5, 'All 11 rows had Tenure = 0\n(brand new customers)\n\n'
                'Total Charges = " " (space) → 0.0\n\n'
                'Why 0? Domain knowledge:\nTenure=0 means never billed,\n'
                'so Total Charges must be 0.\n\nMean imputation ($2,283)\nwould be factually WRONG.',
                transform=axes[0, 1].transAxes, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange'))
axes[0, 1].set_xlim(-1, 1)

# --- Plot 3: Leakage danger — Churn Score vs actual churn ---
raw_df['Churn Score'] = pd.to_numeric(raw_df['Churn Score'], errors='coerce')
raw_df['Churn Value'] = pd.to_numeric(raw_df['Churn Value'], errors='coerce')

for val, color, label in [(0, '#2ecc71', 'Stayed'), (1, '#e74c3c', 'Churned')]:
    subset = raw_df[raw_df['Churn Value'] == val]['Churn Score'].dropna()
    axes[1, 0].hist(subset, bins=30, alpha=0.5, color=color, density=True, label=label)
    subset.plot.kde(ax=axes[1, 0], color=color, linewidth=2)

axes[1, 0].set_title('WHY we dropped Churn Score (LEAKAGE)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Churn Score')
axes[1, 0].legend(fontsize=11)
axes[1, 0].text(0.95, 0.95, 'Churn Score clearly separates\nchurned vs stayed.\n\n'
                'If we kept it, the model would\njust learn "high score = churn"\n'
                'instead of real patterns.\n\nResult: 95%+ accuracy in test,\n'
                'FAILURE in production.',
                transform=axes[1, 0].transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', edgecolor='red'))

# --- Plot 4: Churn Reason — only exists for churned (proof of leakage) ---
reason_counts = raw_df['Churn Reason'].value_counts().head(8)
bars = axes[1, 1].barh(range(len(reason_counts)), reason_counts.values, color='#e74c3c', edgecolor='black')
axes[1, 1].set_yticks(range(len(reason_counts)))
axes[1, 1].set_yticklabels(reason_counts.index, fontsize=9)
axes[1, 1].set_title('WHY we dropped Churn Reason (LEAKAGE)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Count')
axes[1, 1].text(0.95, 0.05, f'Only {raw_df["Churn Reason"].notna().sum()} rows have a reason\n'
                f'= exactly the {int(raw_df["Churn Value"].sum())} churned customers.\n\n'
                'Non-churned customers have no reason\n'
                '→ model learns: "has reason = churned"\n'
                '→ 100% accuracy but completely useless.',
                transform=axes[1, 1].transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', edgecolor='red'))

plt.suptitle('What Data Cleaning Actually Fixed\n(Structural fixes, not distributional — and that\'s correct)',
             fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_11_cleaning_impact.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# STEP 3 SUMMARY
# ============================================================================

print("\n" + "=" * 65)
print(" STEP 3 — DATA CLEANING SUMMARY")
print("=" * 65)
print("""
 What We Did                        | Why This Approach
 -----------------------------------|------------------------------------------
 Removed 1 blank row                | Zero information — nothing to impute from
 Dropped 13 columns                 | Constants, IDs, leakage, high-cardinality
 Total Charges: str → float         | pd.to_numeric(coerce) handles all edge cases
 Filled 11 NaN Total Charges with 0 | Domain knowledge: tenure=0 → never billed
 Kept "No internet service" as-is   | Different churn rate than "No" (~7% vs ~42%)
 Converted target to int            | ML needs numeric targets
 No outlier removal                 | All values within plausible business range
 Kept 22 coincidental feature matches| Different customers, not true duplicates

 WHY distributions barely changed:
   Cleaning here was STRUCTURAL (fix types, remove leakage, fill 11 values),
   not distributional (no outlier removal, no transformations, no imputation
   of large missing chunks). That's CORRECT for this dataset. The real wins
   are invisible in histograms: preventing leakage (fake 95%+ accuracy) and
   fixing types (preventing crashes downstream).

 Result: 7,043 rows × 20 columns, zero missing values
 Saved to: data/processed/telco_churn_cleaned.csv

 NEXT: Step 4 — Feature Engineering
""")
print("=" * 65)
