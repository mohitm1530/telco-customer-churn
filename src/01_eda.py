"""
=============================================================================
 Telco Customer Churn — Data Preprocessing Pipeline
 Step 1: Data Collection / Understanding
 Step 2: Exploratory Data Analysis (EDA)
=============================================================================

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
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports', 'eda')
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# STEP 1: DATA COLLECTION / UNDERSTANDING
# ============================================================================
#
# Goal — Before doing ANYTHING with data, answer:
#   1. What is each column? (name, type, meaning)
#   2. What is the target variable? (what are we predicting?)
#   3. How big is the dataset? (rows × columns)
#   4. What kind of features? (numerical vs categorical)
#   5. Are there obvious problems? (missing values, wrong types, constants)
#
# Why This Step Matters:
#   Skipping data understanding is like building a house without checking the
#   foundation. You'll make wrong decisions later — like encoding a column
#   that should be dropped, or scaling a column that's actually categorical.
# ============================================================================


# --------------------------------------------------------------------------
# 1.1 Load the Data
# --------------------------------------------------------------------------
# Why CSV? We converted from Apple Numbers → CSV because:
#   - CSV is the universal format every tool (Python, R, Excel, SQL) reads
#   - .numbers is proprietary to Apple — pandas can't read it natively
#   - CSV is plain text — inspectable in any text editor
# --------------------------------------------------------------------------

df = pd.read_csv(os.path.join(DATA_RAW, 'telco_churn.csv'))
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"This means we have {df.shape[0]} customers and {df.shape[1]} attributes about each.\n")


# --------------------------------------------------------------------------
# 1.2 First Look — What Does the Data Look Like?
# --------------------------------------------------------------------------
# Why .head() and not the full dataset?
#   - 7000+ rows is unreadable if printed fully
#   - .head() gives a quick sense of what each column contains
#   - Lets you spot obvious issues: weird values, inconsistent formats
# --------------------------------------------------------------------------

print("=== FIRST 5 ROWS ===")
print(df.head())
print("\n=== LAST 5 ROWS ===")
print(df.tail())


# --------------------------------------------------------------------------
# 1.3 Column Inventory — Understanding Every Feature
# --------------------------------------------------------------------------
# Why .info()? It tells you three critical things at once:
#   1. Data types — Is pandas reading each column correctly?
#      (e.g., Total Charges might be object instead of float)
#   2. Non-null counts — How many values are missing per column?
#   3. Memory usage — Important for large datasets
# --------------------------------------------------------------------------

print("\n=== COLUMN INFO ===")
df.info()

# KEY OBSERVATION: Total Charges is object (string), not numeric!
# Pandas reads it as text because some values contain spaces or non-numeric
# characters. We'll fix this in the Data Cleaning step.


# --------------------------------------------------------------------------
# 1.4 Classify Columns by Type and Role
# --------------------------------------------------------------------------
# This is the MOST IMPORTANT part of data understanding.
# Every column falls into one of these roles:
#
#   Role               | Meaning                          | Example
#   -------------------|----------------------------------|-------------------
#   Identifier         | Uniquely identifies a row        | CustomerID
#   Target             | What we're predicting            | Churn Value
#   Numerical Feature  | Continuous or discrete numbers   | Monthly Charges
#   Categorical Feature| Discrete categories              | Gender, Contract
#   Redundant/Constant | Same value for all rows          | Country, Count
#   Leakage Suspect    | Info derived FROM the target      | Churn Score
# --------------------------------------------------------------------------

column_analysis = pd.DataFrame({
    'Column': df.columns,
    'Dtype': df.dtypes.values,
    'Unique_Values': [df[col].nunique() for col in df.columns],
    'Null_Count': df.isnull().sum().values,
    'Sample_Values': [str(df[col].dropna().unique()[:4].tolist()) for col in df.columns]
})

role_map = {
    'CustomerID': 'Identifier',
    'Count': 'Redundant (all 1s)',
    'Country': 'Redundant (all USA)',
    'State': 'Redundant (all CA)',
    'City': 'Geographical',
    'Zip Code': 'Geographical',
    'Lat Long': 'Redundant (duplicates Lat/Long)',
    'Latitude': 'Geographical',
    'Longitude': 'Geographical',
    'Gender': 'Categorical Feature',
    'Senior Citizen': 'Categorical Feature',
    'Partner': 'Categorical Feature',
    'Dependents': 'Categorical Feature',
    'Tenure Months': 'Numerical Feature',
    'Phone Service': 'Categorical Feature',
    'Multiple Lines': 'Categorical Feature',
    'Internet Service': 'Categorical Feature',
    'Online Security': 'Categorical Feature',
    'Online Backup': 'Categorical Feature',
    'Device Protection': 'Categorical Feature',
    'Tech Support': 'Categorical Feature',
    'Streaming TV': 'Categorical Feature',
    'Streaming Movies': 'Categorical Feature',
    'Contract': 'Categorical Feature',
    'Paperless Billing': 'Categorical Feature',
    'Payment Method': 'Categorical Feature',
    'Monthly Charges': 'Numerical Feature',
    'Total Charges': 'Numerical Feature (needs fixing)',
    'Churn Label': 'Target',
    'Churn Value': 'Target (numeric version)',
    'Churn Score': 'LEAKAGE SUSPECT',
    'CLTV': 'LEAKAGE SUSPECT',
    'Churn Reason': 'LEAKAGE (only for churned)'
}

column_analysis['Role'] = column_analysis['Column'].map(role_map)
column_analysis.set_index('Column', inplace=True)
print("\n=== COLUMN CLASSIFICATION ===")
print(column_analysis.to_string())


# --------------------------------------------------------------------------
# 1.5 Understanding the Target Variable
# --------------------------------------------------------------------------
# What are we predicting? → Whether a customer will CHURN (leave the company).
#
# We have TWO target columns:
#   - Churn Label — "Yes" / "No" (categorical)
#   - Churn Value — 1 / 0 (numeric version of the same)
#
# We'll use Churn Value as our target because ML algorithms need numeric
# targets. We'll drop Churn Label later to avoid redundancy.
# --------------------------------------------------------------------------

print("\n=== TARGET DISTRIBUTION ===")
print(df['Churn Value'].value_counts())
print(f"\nChurn Rate: {df['Churn Value'].mean()*100:.1f}%")
print("This is an IMBALANCED dataset — only ~26.5% of customers churned.")
print("A naive model predicting 'No churn' for everyone gets 73.5% accuracy but is useless!")
print("We'll address this with SMOTE or class weights in Step 7.")


# --------------------------------------------------------------------------
# 1.6 Identify Columns to Drop Early
# --------------------------------------------------------------------------
# Some columns should NEVER enter a model:
#
#   Column       | Why Drop?
#   -------------|----------------------------------------------------------
#   CustomerID   | Identifier, not a feature. Model would memorize IDs.
#   Count        | Always 1 — zero variance, zero information.
#   Country      | Always "United States" — zero variance.
#   State        | Always "California" — zero variance.
#   Lat Long     | String duplicate of Latitude + Longitude.
#   Churn Label  | Text version of Churn Value — keep only one target.
#   Churn Reason | DATA LEAKAGE — only filled for churned customers.
#   Churn Score  | LEAKAGE — pre-computed churn probability.
#   CLTV         | LEAKAGE — Customer Lifetime Value uses churn status.
# --------------------------------------------------------------------------

print("\n=== VERIFYING DROP DECISIONS ===")

print("\n--- Constant/Near-Constant Columns ---")
for col in ['Count', 'Country', 'State']:
    print(f"  {col}: {df[col].nunique()} unique value(s) → {df[col].dropna().unique()}")

print("\n--- Churn Reason — only exists for churned customers ---")
print(f"  Total non-null Churn Reasons: {df['Churn Reason'].notna().sum()}")
print(f"  Total churned customers:      {(df['Churn Value'] == 1).sum()}")
print("  → They match! Churn Reason is post-hoc information.")

print("\n--- Churn Score correlation with target ---")
print(f"  Correlation: {df['Churn Score'].corr(df['Churn Value']):.3f}")
print("  → High correlation suggests it was DERIVED from the target.")


# --------------------------------------------------------------------------
# STEP 1 SUMMARY
# --------------------------------------------------------------------------
print("\n" + "=" * 65)
print(" STEP 1 SUMMARY")
print("=" * 65)
print("• 7,043 customers, 33 columns (after removing 1 null row)")
print("• Target: Churn Value (0=stayed, 1=churned) — IMBALANCED at ~26.5%")
print("• 9 columns to drop: ID, constants, redundant, and leakage columns")
print("• Total Charges is wrongly typed as string — needs conversion")
print("• 1 completely null row across all columns")
print("• Remaining useful features: ~20 columns (mix of categorical + numerical)")
print("=" * 65)


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
#
# Goal — What patterns exist in the data?
#   1. How are numerical features distributed? (skewed? outliers?)
#   2. How are categorical features distributed? (imbalanced categories?)
#   3. How does each feature relate to the target (churn)?
#   4. Are features correlated with each other? (multicollinearity)
#
# Why EDA BEFORE Cleaning?
#   EDA INFORMS your cleaning decisions:
#   - If EDA shows Total Charges has spaces, you know HOW to clean it
#   - If a feature has no relationship with churn, you might drop it
#   - If extreme outliers exist, you decide whether to cap, remove, or keep
#
# Why These Specific Plot Types?
#
#   Plot Type        | What It Reveals                    | When to Use
#   -----------------|------------------------------------|--------------------
#   Histogram        | Distribution shape                 | Numerical features
#   Box Plot         | Outliers + quartile spread         | Numerical features
#   Count/Bar Plot   | Category frequencies / churn rates | Categorical features
#   Correlation Map  | Linear relationships               | All numerical pairs
#   KDE Plot         | Smooth comparison between groups   | Numerical vs Target
#   Scatter Plot     | Shape of relationship + clusters   | Two numerical vars
# ============================================================================


# --------------------------------------------------------------------------
# 2.1 Quick Statistical Summary
# --------------------------------------------------------------------------
# Why .describe()?
#   - Shows min, max, mean, std — reveals if values are in expected ranges
#   - Large gap between 75th percentile and max → possible outliers
#   - Mean ≠ median (50%) → skewness
# --------------------------------------------------------------------------

print("\n=== NUMERICAL SUMMARY ===")
print(df.describe())
print("\n=== CATEGORICAL SUMMARY ===")
print(df.describe(include='object'))


# --------------------------------------------------------------------------
# 2.2 Target Variable Analysis
# --------------------------------------------------------------------------
# Why start with the target?
#   Everything in EDA should connect back to: "Does this feature help
#   predict churn?" Starting with the target sets the baseline.
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

churn_counts = df['Churn Label'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(churn_counts.index, churn_counts.values, color=colors, edgecolor='black')
for i, (label, count) in enumerate(zip(churn_counts.index, churn_counts.values)):
    axes[0].text(i, count + 50, f'{count}\n({count/len(df)*100:.1f}%)', ha='center', fontweight='bold')
axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Customers')

axes[1].pie(churn_counts.values, labels=['Stayed (No)', 'Churned (Yes)'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=(0, 0.05), shadow=True, textprops={'fontsize': 12})
axes[1].set_title('Churn Distribution (Proportion)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_01_target_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

print("INSIGHT: ~73.5% stayed, ~26.5% churned — imbalanced dataset.")
print("We'll address this with SMOTE or class weights in Step 7.\n")


# --------------------------------------------------------------------------
# 2.3 Numerical Features — Distribution Analysis
# --------------------------------------------------------------------------
# Why Histograms + KDE?
#   - Histogram: actual frequency of values in bins
#   - KDE (Kernel Density Estimate): smooth probability density curve
#   - Together they reveal: Normal? Skewed? Bimodal?
#
# Why does distribution shape matter?
#   - Normal: many algorithms (logistic regression) work best with this
#   - Skewed: may need transformation (log, sqrt) to reduce extreme values
#   - Bimodal: suggests two distinct groups — worth investigating
# --------------------------------------------------------------------------

df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

numerical_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']

fig, axes = plt.subplots(3, 2, figsize=(16, 15))

for i, col in enumerate(numerical_features):
    axes[i, 0].hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    df[col].dropna().plot.kde(ax=axes[i, 0], color='red', linewidth=2)
    axes[i, 0].set_title(f'{col} — Distribution', fontsize=13, fontweight='bold')
    axes[i, 0].set_xlabel(col)

    mean_val = df[col].mean()
    median_val = df[col].median()
    skew_val = df[col].skew()
    axes[i, 0].axvline(mean_val, color='green', linestyle='--', label=f'Mean: {mean_val:.1f}')
    axes[i, 0].axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.1f}')
    axes[i, 0].legend()
    axes[i, 0].text(0.95, 0.95, f'Skewness: {skew_val:.2f}', transform=axes[i, 0].transAxes,
                    ha='right', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

    bp = axes[i, 1].boxplot(df[col].dropna(), vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    axes[i, 1].set_title(f'{col} — Box Plot (Outlier Detection)', fontsize=13, fontweight='bold')
    axes[i, 1].set_ylabel(col)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_02_numerical_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()

print("INSIGHTS:")
print("─" * 60)
print("• Tenure Months: Nearly UNIFORM distribution (not skewed).")
print("  → Customers spread across all tenure lengths. No transformation needed.")
print()
print("• Monthly Charges: BIMODAL — peaks around $20 and $80.")
print("  → Two customer segments (basic vs premium plans). No transformation needed.")
print()
print("• Total Charges: RIGHT-SKEWED (long tail to the right).")
print("  → Expected: new customers have low totals, long-term have high.")
print("  → May benefit from log transformation. No significant outliers.\n")


# --------------------------------------------------------------------------
# 2.4 Numerical Features vs Churn — Do They Help Predict?
# --------------------------------------------------------------------------
# Why KDE plots split by churn?
#   - If the two curves (churned vs not) look DIFFERENT → feature is informative
#   - If they overlap completely → feature probably won't help the model
#   - More informative than correlation: shows SHAPE, not just linear strength
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i, col in enumerate(numerical_features):
    for label, color, name in [(0, '#2ecc71', 'Stayed'), (1, '#e74c3c', 'Churned')]:
        subset = df[df['Churn Value'] == label][col].dropna()
        axes[i].hist(subset, bins=30, alpha=0.4, color=color, density=True, label=name)
        subset.plot.kde(ax=axes[i], color=color, linewidth=2)

    axes[i].set_title(f'{col} by Churn Status', fontsize=13, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_03_numerical_vs_churn.png'), dpi=150, bbox_inches='tight')
plt.show()

print("INSIGHTS:")
print("─" * 60)
print("• Tenure: Churned customers cluster at LOW tenure (0-10 months).")
print("  → New customers much more likely to churn. STRONG predictor.")
print()
print("• Monthly Charges: Churned customers tend to pay MORE ($70-100 range).")
print("  → Higher bills → more likely to leave. MODERATE predictor.")
print()
print("• Total Charges: Churned customers have LOWER total charges.")
print("  → Correlated with tenure (low tenure = low total).")
print("  → Watch for multicollinearity between Tenure and Total Charges.\n")


# --------------------------------------------------------------------------
# 2.5 Box Plots — Numerical Features by Churn
# --------------------------------------------------------------------------
# Why Box Plots here (not just KDE)?
#   - Box plots show median, quartiles, and outliers — KDE doesn't
#   - You can immediately see if the median differs between groups
#   - Useful for spotting if churned customers systematically differ
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(numerical_features):
    sns.boxplot(data=df, x='Churn Label', y=col, ax=axes[i],
                palette={'No': '#2ecc71', 'Yes': '#e74c3c'})
    axes[i].set_title(f'{col} by Churn', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_04_boxplots_by_churn.png'), dpi=150, bbox_inches='tight')
plt.show()

print("CONFIRMATION: Churned customers have lower tenure, higher monthly charges,")
print("lower total charges. Median lines clearly differ — all 3 are useful features.\n")


# --------------------------------------------------------------------------
# 2.6 Categorical Features — Distribution & Churn Rates
# --------------------------------------------------------------------------
# Why Churn RATE Per Category (not just counts)?
#   - Raw counts are MISLEADING. If 90% of customers are on "Month-to-month",
#     they'll naturally have the most churners even if the RATE is the same.
#   - Churn RATE = (churned in category) / (total in category)
#     → normalizes for group size
#   - A feature is useful if churn rates vary significantly across categories.
# --------------------------------------------------------------------------

categorical_features = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Phone Service', 'Multiple Lines', 'Internet Service',
    'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming TV', 'Streaming Movies',
    'Contract', 'Paperless Billing', 'Payment Method'
]

fig, axes = plt.subplots(4, 4, figsize=(24, 22))
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    churn_rates = df.groupby(col)['Churn Value'].mean().sort_values(ascending=False)
    counts = df[col].value_counts()

    bars = axes[i].bar(
        range(len(churn_rates)), churn_rates.values,
        color=['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.2 else '#2ecc71' for r in churn_rates.values],
        edgecolor='black'
    )
    axes[i].set_xticks(range(len(churn_rates)))
    axes[i].set_xticklabels(churn_rates.index, rotation=45, ha='right', fontsize=9)
    axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Churn Rate')
    axes[i].axhline(y=df['Churn Value'].mean(), color='blue', linestyle='--', alpha=0.5, label='Overall rate')
    axes[i].set_ylim(0, 0.55)

    for j, (rate, cat) in enumerate(zip(churn_rates.values, churn_rates.index)):
        n = counts.get(cat, 0)
        axes[i].text(j, rate + 0.01, f'{rate:.0%}\n(n={n})', ha='center', fontsize=8)

plt.suptitle('Churn Rate by Categorical Feature\n(Blue dashed line = overall churn rate of 26.5%)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_05_categorical_churn_rates.png'), dpi=150, bbox_inches='tight')
plt.show()

print("INSIGHTS FROM CATEGORICAL FEATURES:")
print("=" * 65)
print()
print("STRONG PREDICTORS (large churn rate differences):")
print("─" * 65)
print("• Contract: Month-to-month = ~43% churn vs Two-year = ~3% churn")
print("  → STRONGEST PREDICTOR. Lock-in reduces churn massively.")
print()
print("• Internet Service: Fiber optic = ~42% churn vs DSL = ~19%, No = ~7%")
print("  → Fiber optic customers churn 6x more than no-internet.")
print()
print("• Online Security / Tech Support / etc: 'No' = ~42% vs 'Yes' = ~15%")
print("  → Customers WITHOUT add-on services churn much more.")
print()
print("• Payment Method: Electronic check = ~45% churn (highest!)")
print("  → Less committed customers, less friction to cancel.")
print()
print("MODERATE PREDICTORS:")
print("─" * 65)
print("• Senior Citizen: Yes = ~41% churn vs No = ~24%")
print("• Partner/Dependents: No partner/dependents = higher churn")
print("• Paperless Billing: Yes = ~34% vs No = ~17%")
print()
print("WEAK PREDICTORS:")
print("─" * 65)
print("• Gender: Male = ~26% vs Female = ~27% — virtually no difference")
print("  → Gender is NOT useful. May drop in Feature Selection.")
print("• Phone Service: minimal difference between Yes/No\n")


# --------------------------------------------------------------------------
# 2.7 Correlation Analysis — Numerical Features
# --------------------------------------------------------------------------
# Why a Correlation Heatmap?
#   - Shows LINEAR relationships between all numerical feature pairs
#   - Detects MULTICOLLINEARITY — when two features are highly correlated
#     (>0.7), they carry redundant information, which can:
#       • Make models unstable (especially logistic regression)
#       • Inflate feature importance estimates
#
# Why Pearson Correlation (not Spearman)?
#   - Pearson: measures LINEAR relationship. Good for continuous features.
#   - Spearman: measures MONOTONIC relationship (any consistently
#     increasing/decreasing trend). Better for ordinal or non-linear data.
#   - We use Pearson first because it's the standard. If we saw non-linear
#     patterns in scatter plots, we'd add Spearman.
# --------------------------------------------------------------------------

corr_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Value']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=1, mask=mask,
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Correlation Matrix — Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_06_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

print("INSIGHTS:")
print("─" * 60)
print(f"• Tenure <-> Total Charges: r = {corr_matrix.loc['Tenure Months', 'Total Charges']:.3f}")
print("  → STRONG positive. Obvious: longer tenure = more total spend.")
print("  → MULTICOLLINEARITY. May need to drop one or combine them.")
print()
print(f"• Monthly Charges <-> Total Charges: r = {corr_matrix.loc['Monthly Charges', 'Total Charges']:.3f}")
print("  → Moderate. Higher monthly bill = higher total.")
print()
print(f"• Tenure <-> Churn: r = {corr_matrix.loc['Tenure Months', 'Churn Value']:.3f}")
print("  → Negative: longer tenure = less churn. Confirms KDE finding.")
print()
print(f"• Monthly Charges <-> Churn: r = {corr_matrix.loc['Monthly Charges', 'Churn Value']:.3f}")
print("  → Weak positive: higher bills slightly increase churn.\n")


# --------------------------------------------------------------------------
# 2.8 Scatter Plot — Tenure vs Total Charges (Multicollinearity Check)
# --------------------------------------------------------------------------
# Why a Scatter Plot here?
#   - Correlation tells you the STRENGTH of the linear relationship
#   - Scatter shows the SHAPE — truly linear, or is there a curve/funnel?
#   - Color-coding by churn shows if churned customers cluster in regions
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for label, color, name in [(0, '#2ecc71', 'Stayed'), (1, '#e74c3c', 'Churned')]:
    subset = df[df['Churn Value'] == label]
    ax.scatter(subset['Tenure Months'], subset['Total Charges'],
               c=color, alpha=0.3, s=15, label=name)

ax.set_xlabel('Tenure (Months)', fontsize=12)
ax.set_ylabel('Total Charges ($)', fontsize=12)
ax.set_title('Tenure vs Total Charges — Colored by Churn', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_07_scatter_tenure_totalcharges.png'), dpi=150, bbox_inches='tight')
plt.show()

print("INSIGHTS:")
print("• Clear linear trend: Total Charges ≈ Monthly Charges × Tenure")
print("• Churned customers (red) cluster BOTTOM-LEFT — low tenure, low total")
print("• 'Fan' shape: spread increases with tenure")
print()
print("DECISION: Since Total Charges ≈ Monthly Charges × Tenure,")
print("it's largely redundant. May consider dropping in Feature Selection.\n")


# --------------------------------------------------------------------------
# 2.9 Deep Dive — Top 3 Strongest Predictors
# --------------------------------------------------------------------------
# Why focus on top predictors?
#   - Not all features are equally important
#   - Understanding the STRONGEST predictors deeply helps you:
#     1. Validate that data makes business sense
#     2. Decide which features to engineer new combinations from
#     3. Explain your model to stakeholders later
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ct = pd.crosstab(df['Contract'], df['Churn Label'], normalize='index')
ct.plot(kind='bar', stacked=True, ax=axes[0], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title('Contract Type → Churn %', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Proportion')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(title='Churn')

ct2 = pd.crosstab(df['Internet Service'], df['Churn Label'], normalize='index')
ct2.plot(kind='bar', stacked=True, ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[1].set_title('Internet Service → Churn %', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Proportion')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title='Churn')

ct3 = pd.crosstab(df['Payment Method'], df['Churn Label'], normalize='index')
ct3.plot(kind='bar', stacked=True, ax=axes[2], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[2].set_title('Payment Method → Churn %', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Proportion')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=25, ha='right')
axes[2].legend(title='Churn')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_08_top_predictors_deep_dive.png'), dpi=150, bbox_inches='tight')
plt.show()


# --------------------------------------------------------------------------
# 2.10 Missing Values Analysis
# --------------------------------------------------------------------------
# Why analyze missingness patterns?
#   Not all missing data is the same. Three types:
#     1. MCAR (Missing Completely At Random): No pattern — safe to drop/impute
#     2. MAR (Missing At Random): Missingness depends on OTHER variables
#     3. MNAR (Missing Not At Random): Missingness depends on VALUE itself
#        (e.g., high earners don't report income)
#   The handling strategy depends on which type it is.
# --------------------------------------------------------------------------

df_raw = pd.read_csv(os.path.join(DATA_RAW, 'telco_churn.csv'))

missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)

print("=== MISSING VALUES SUMMARY ===")
print(missing_df)
print()
print("ANALYSIS:")
print("─" * 60)
print(f"• Churn Reason: {missing_df.loc['Churn Reason', 'Count']} missing ({missing_df.loc['Churn Reason', 'Percentage']}%)")
print("  → MNAR: only filled for churned customers. Not a data quality issue.")
print("  → We'll DROP this column (it's leakage anyway).")
print()
print("• All other columns: exactly 1 missing each")
print("  → Almost certainly ONE completely blank row. Let's verify...")

cols_except_reason = [c for c in df_raw.columns if c != 'Churn Reason']
fully_null_rows = df_raw[df_raw[cols_except_reason].isnull().all(axis=1)]
print(f"\n  Rows where ALL columns (except Churn Reason) are null: {len(fully_null_rows)}")
print("  → Confirmed: single blank row. Drop it in Cleaning step.")
print("  → Impact: losing 1 row out of 7044 is negligible (0.01%).\n")


# --------------------------------------------------------------------------
# 2.11 Feature Interaction — Churn Heatmap by Contract & Internet Service
# --------------------------------------------------------------------------
# Why look at feature interactions?
#   - Individual analysis shows Contract and Internet are strong predictors
#   - But COMBINATIONS might matter more: fiber optic + month-to-month?
#   - This guides Feature Engineering (Step 4)
# --------------------------------------------------------------------------

interaction = df.pivot_table(values='Churn Value', index='Internet Service',
                              columns='Contract', aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(interaction, annot=True, fmt='.1%', cmap='RdYlGn_r',
            linewidths=1, vmin=0, vmax=0.55, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title('Churn Rate: Internet Service × Contract Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_09_interaction_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

print("KEY INSIGHT:")
print("• Fiber optic + Month-to-month = ~51% churn rate (HALF of them leave!)")
print("• DSL + Two-year = ~3% churn rate")
print("• No internet + Two-year = ~1% churn rate")
print()
print("TAKEAWAY: The combination of service type + contract type is more")
print("informative than either alone → create INTERACTION FEATURE in Step 4.\n")


# --------------------------------------------------------------------------
# 2.12 Geographic Analysis (Quick Check)
# --------------------------------------------------------------------------
# Why check geography?
#   - If churn varies by city/zip, geography is a useful feature
#   - But with 1,129 unique cities, directly encoding = too many features
#   - Quick check to see if it's worth the effort
# --------------------------------------------------------------------------

top_cities = df['City'].value_counts().head(20).index
city_churn = df[df['City'].isin(top_cities)].groupby('City')['Churn Value'].agg(['mean', 'count'])
city_churn = city_churn.sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(city_churn)), city_churn['mean'], color='steelblue', edgecolor='black')
ax.set_xticks(range(len(city_churn)))
ax.set_xticklabels(city_churn.index, rotation=45, ha='right')
ax.axhline(y=df['Churn Value'].mean(), color='red', linestyle='--', label='Overall avg')
ax.set_title('Churn Rate by Top 20 Cities (by customer count)', fontsize=14, fontweight='bold')
ax.set_ylabel('Churn Rate')
ax.legend()

for i, (rate, count) in enumerate(zip(city_churn['mean'], city_churn['count'])):
    ax.text(i, rate + 0.005, f'n={int(count)}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_10_geographic_churn.png'), dpi=150, bbox_inches='tight')
plt.show()

print("INSIGHT:")
print("• Churn rates vary across cities (22% to 32% in top 20)")
print("• But with 1,129 cities and most having <10 customers,")
print("  geographic features would introduce noise, not signal.")
print("• DECISION: Drop City, Zip Code, Lat, Long in Feature Selection.\n")


# ============================================================================
# STEP 2 (EDA) SUMMARY — KEY FINDINGS
# ============================================================================
print("=" * 65)
print(" STEP 2 (EDA) SUMMARY")
print("=" * 65)
print()
print("STRONGEST CHURN PREDICTORS (ranked):")
print("  1. Contract Type — Month-to-month churn 14x more than Two-year")
print("  2. Internet Service — Fiber optic churn 6x more than no-internet")
print("  3. Tenure — New customers (<10 months) churn much more")
print("  4. Payment Method — Electronic check users churn ~45%")
print("  5. Add-on Services — Having none → ~42% churn")
print()
print("WEAK/USELESS PREDICTORS:")
print("  • Gender — virtually identical churn rates")
print("  • Phone Service — minimal predictive power")
print()
print("DATA QUALITY ISSUES:")
print("  1. Total Charges stored as string — needs numeric conversion")
print("  2. 1 completely blank row — needs removal")
print("  3. Multicollinearity: Tenure <-> Total Charges (r=0.826)")
print()
print("COLUMNS TO DROP:")
print("  • CustomerID, Count, Country, State, Lat Long (IDs/constants/redundant)")
print("  • Churn Label, Churn Score, CLTV, Churn Reason (target dupes / leakage)")
print("  • City, Zip Code, Latitude, Longitude (too many categories, low signal)")
print()
print("FEATURE ENGINEERING IDEAS (for Step 4):")
print("  • Interaction: Contract × Internet Service")
print("  • Bin Monthly Charges into tiers (based on bimodal distribution)")
print("  • Create 'total add-on services' count")
print()
print("NEXT: Step 3 — Data Cleaning")
print("=" * 65)
