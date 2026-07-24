"""
=============================================================================
 Telco Customer Churn — Data Preprocessing Pipeline
 Step 4: Feature Engineering
=============================================================================

Feature Engineering = creating NEW features from existing ones to give the
model more useful signals to learn from.

WHY do we need it?
  Raw features are what the database stores. But the PATTERNS that predict
  churn often live in COMBINATIONS and TRANSFORMATIONS of raw features.
  Example: "Tenure Months" and "Monthly Charges" are raw features.
  But "Average Monthly Charges relative to tenure" captures a richer pattern
  that neither raw feature alone can express.

WHAT makes a good engineered feature?
  1. It captures domain knowledge (business logic the model can't discover alone)
  2. It reduces complexity (e.g., counting add-ons instead of 6 separate binary columns)
  3. It exposes non-linear relationships (e.g., binning tenure into groups)
  4. It creates interactions the model might miss (e.g., Contract × Internet type)

ORDER OF OPERATIONS:
  Feature engineering happens BEFORE encoding and scaling because:
  - We need the original categorical values to create meaningful combinations
  - Binning and grouping work on raw numeric scales
  - Encoding (Step 5) should happen on the FINAL set of features
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
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports', 'feature_engineering')
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# LOAD CLEANED DATA
# ============================================================================

df = pd.read_csv(os.path.join(DATA_PROCESSED, 'telco_churn_cleaned.csv'))
print(f"Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")


# ============================================================================
# 4.1 TENURE BINS — Discretizing Continuous Tenure
# ============================================================================
#
# WHY bin tenure instead of keeping it continuous?
#
#   EDA showed that churn risk is NOT linearly proportional to tenure.
#   Instead, it drops sharply in the first year, then gradually flattens:
#     - 0-12 months:  ~47% churn (very high risk)
#     - 13-24 months: ~30% churn (moderate risk)
#     - 25-48 months: ~22% churn (declining)
#     - 49-72 months: ~12% churn (loyal customers)
#
#   A linear model using raw tenure would try to fit a straight line through
#   this curve and miss the sharp early drop-off. Bins capture this non-linear
#   step-function pattern directly.
#
# WHY these specific bin edges (not equal-width or quantile)?
#
#   Option A: Equal-width bins (0-18, 18-36, 36-54, 54-72)
#     → Bad because the critical early churn period (0-12 months) gets mixed
#       with the stabilizing period (12-18 months). You lose the signal.
#
#   Option B: Equal-frequency / quantile bins (each bin has ~25% of data)
#     → Reasonable, but ignores domain knowledge. The business cares about
#       specific milestones: first year (trial period), second year, etc.
#
#   Option C: Domain-driven bins (our choice)
#     → 0-12: first year = trial/onboarding period
#     → 13-24: second year = settling in
#     → 25-48: mid-term = established customer
#     → 49-72: long-term = loyal/locked-in
#     These align with business lifecycle stages AND match the churn rate
#     breakpoints we saw in EDA.
#
# WHY keep raw tenure AND add bins?
#   - Raw tenure gives the model fine-grained linear signal
#   - Bins give the model a non-linear step-function signal
#   - Tree-based models (Random Forest, XGBoost) can learn non-linear
#     splits from raw tenure, so bins help them less
#   - Linear models (Logistic Regression) CAN'T learn non-linear splits,
#     so bins are critical for them
#   - Keeping both lets any model type benefit
# ============================================================================

print("4.1 — Creating Tenure Bins")

bins = [0, 12, 24, 48, 72]
labels = ['0-12 mo', '13-24 mo', '25-48 mo', '49-72 mo']
df['Tenure Group'] = pd.cut(df['Tenure Months'], bins=bins, labels=labels, include_lowest=True)

churn_by_tenure = df.groupby('Tenure Group', observed=True)['Churn Value'].agg(['mean', 'count'])
print("     Churn rate by tenure group:")
for group, row in churn_by_tenure.iterrows():
    print(f"       {group}: {row['mean']:.1%} churn (n={int(row['count'])})")

print()


# ============================================================================
# 4.2 ADD-ON SERVICE COUNT
# ============================================================================
#
# WHY create a service count instead of using individual columns?
#
#   We have 6 add-on service columns (Online Security, Online Backup, Device
#   Protection, Tech Support, Streaming TV, Streaming Movies). Each is "Yes",
#   "No", or "No internet service".
#
#   Option A: Keep all 6 individual columns
#     → After one-hot encoding, this creates 12-18 binary columns (2-3 per feature)
#     → EDA showed all 6 have SIMILAR churn patterns: "Yes" = ~15% churn,
#       "No" = ~42% churn. They all tell roughly the same story.
#     → High redundancy among these features increases multicollinearity.
#
#   Option B: Replace all 6 with a single count (our choice)
#     → Captures "depth of product penetration" — how many services a customer
#       uses. This is a well-established churn predictor in telecom literature.
#     → Reduces 6 correlated columns to 1 informative column
#     → Research (SHAP analyses) shows that total service count is often
#       ranked higher in feature importance than individual service flags.
#     → A customer with 5 add-ons is much less likely to churn than one
#       with 0, and this single number captures that gradient.
#
#   Option C: Create count AND keep individual columns
#     → Risks multicollinearity: the count is a linear combination of the
#       individuals. Logistic regression would struggle with this.
#     → Only useful if different add-ons have very different churn impacts
#       (they don't in our data — all ~15% vs ~42%).
#
#   We keep the individual columns for now and let Feature Selection (Step 5)
#   decide whether to drop them based on statistical tests.
# ============================================================================

print("4.2 — Creating Add-on Service Count")

addon_cols = ['Online Security', 'Online Backup', 'Device Protection',
              'Tech Support', 'Streaming TV', 'Streaming Movies']

df['Num Addon Services'] = df[addon_cols].apply(
    lambda row: (row == 'Yes').sum(), axis=1
)

churn_by_addons = df.groupby('Num Addon Services')['Churn Value'].agg(['mean', 'count'])
print("     Churn rate by number of add-on services:")
for n_addons, row in churn_by_addons.iterrows():
    print(f"       {n_addons} add-ons: {row['mean']:.1%} churn (n={int(row['count'])})")

print()


# ============================================================================
# 4.3 MONTHLY CHARGES TIER
# ============================================================================
#
# WHY bin Monthly Charges?
#
#   EDA revealed a BIMODAL distribution — peaks around $20 and $80.
#   This suggests two natural customer segments:
#     - Low tier (~$18-35): basic plans, fewer services
#     - Mid tier (~$35-70): moderate plans
#     - High tier (~$70-120): premium plans with multiple services
#
#   Churn rates differ across these tiers (higher charges → higher churn).
#   Bins capture this segmentation explicitly.
#
# WHY these specific edges?
#   → Based on the bimodal distribution from EDA:
#     $18-35 = the first peak (basic plan customers)
#     $35-70 = the valley between peaks
#     $70-120 = the second peak (premium customers)
#   → These aren't arbitrary — they follow the natural clusters in the data.
#
# WHY NOT use quantile bins here?
#   → Quantile bins would split the data into equal-sized groups, but
#     they'd cut through the natural peaks in the distribution.
#     The bimodal shape means the business has natural pricing tiers,
#     and our bins should respect those tiers.
# ============================================================================

print("4.3 — Creating Monthly Charges Tier")

charge_bins = [0, 35, 70, 120]
charge_labels = ['Low ($18-35)', 'Mid ($35-70)', 'High ($70-120)']
df['Charges Tier'] = pd.cut(df['Monthly Charges'], bins=charge_bins,
                            labels=charge_labels, include_lowest=True)

churn_by_tier = df.groupby('Charges Tier', observed=True)['Churn Value'].agg(['mean', 'count'])
print("     Churn rate by charges tier:")
for tier, row in churn_by_tier.iterrows():
    print(f"       {tier}: {row['mean']:.1%} churn (n={int(row['count'])})")

print()


# ============================================================================
# 4.4 CONTRACT-INTERNET INTERACTION
# ============================================================================
#
# WHY create interaction features?
#
#   EDA (Section 2.11) showed that the COMBINATION of Contract type and
#   Internet Service is far more predictive than either alone:
#     - Fiber optic + Month-to-month = 51% churn
#     - DSL + Two-year = 3% churn
#     - No internet + Two-year = 1% churn
#
#   Linear models can only learn ADDITIVE effects by default:
#     churn ~ β₁×Contract + β₂×InternetService
#   They CANNOT learn that the effect of Contract DEPENDS on Internet type
#   unless we explicitly create the interaction term:
#     churn ~ β₁×Contract + β₂×Internet + β₃×(Contract × Internet)
#
# WHY string concatenation (not polynomial features)?
#
#   Option A: sklearn PolynomialFeatures
#     → Works on NUMERIC features. Our Contract and Internet Service are
#       categorical, so we'd need to encode first, then multiply columns.
#       This creates many unintuitive columns (e.g., Contract_MTM × Internet_DSL).
#
#   Option B: String concatenation (our choice)
#     → Creates a new categorical column: "Month-to-month_Fiber optic"
#     → Intuitive, directly interpretable
#     → Gets one-hot encoded later like any other categorical
#     → Each combination gets its own coefficient in the model
#
#   Option C: Target encoding the interaction
#     → Replace each combination with its mean churn rate
#     → Risks target leakage if not done carefully with cross-validation
#     → More complex than needed for ~9 unique combinations
# ============================================================================

print("4.4 — Creating Contract-Internet Interaction")

df['Contract_Internet'] = df['Contract'] + '_' + df['Internet Service']

interaction_churn = df.groupby('Contract_Internet')['Churn Value'].agg(['mean', 'count'])
interaction_churn = interaction_churn.sort_values('mean', ascending=False)
print("     Churn rate by Contract × Internet Service:")
for combo, row in interaction_churn.iterrows():
    print(f"       {combo}: {row['mean']:.1%} churn (n={int(row['count'])})")

print()


# ============================================================================
# 4.5 AVERAGE MONTHLY SPEND (Total Charges / Tenure)
# ============================================================================
#
# WHY create this ratio?
#
#   Total Charges = Monthly Charges × Tenure (approximately).
#   But real spending patterns aren't perfectly linear — customers may
#   upgrade/downgrade plans, get promotional rates, etc.
#
#   Average Monthly Spend = Total Charges / Tenure captures the TRUE
#   average monthly rate including any plan changes, promos, or billing
#   adjustments that happened over the customer's lifetime.
#
#   It differs from "Monthly Charges" (the CURRENT plan price) because:
#   - A customer currently paying $80/month who started at $30/month
#     has a lower average spend than current charges suggest
#   - This gap itself could indicate price sensitivity (rapid price hikes)
#
# WHY handle the Tenure=0 case specially?
#   → Division by zero. For Tenure=0 customers, we use their Monthly Charges
#     as the average (they haven't been billed yet, so current rate = only
#     data point we have).
# ============================================================================

print("4.5 — Creating Average Monthly Spend")

df['Avg Monthly Spend'] = np.where(
    df['Tenure Months'] == 0,
    df['Monthly Charges'],
    df['Total Charges'] / df['Tenure Months']
)

print(f"     Range: ${df['Avg Monthly Spend'].min():.2f} — ${df['Avg Monthly Spend'].max():.2f}")
print(f"     Mean:  ${df['Avg Monthly Spend'].mean():.2f}")
print(f"     Correlation with Monthly Charges: {df['Avg Monthly Spend'].corr(df['Monthly Charges']):.3f}")
print(f"     Correlation with Churn: {df['Avg Monthly Spend'].corr(df['Churn Value']):.3f}")
print()


# ============================================================================
# 4.6 ENGAGEMENT SCORE — Multi-factor Loyalty Indicator
# ============================================================================
#
# WHY create a composite engagement score?
#
#   Individual features capture different DIMENSIONS of customer engagement:
#     - Tenure: how LONG they've stayed (loyalty over time)
#     - Monthly Charges: how much they SPEND (financial commitment)
#     - Num Addon Services: how deeply they USE the product (stickiness)
#
#   No single feature captures overall engagement. A customer with:
#     - 60 months tenure + $20/month + 0 add-ons → long but disengaged
#     - 6 months tenure + $100/month + 5 add-ons → short but deeply invested
#   These have very different churn profiles that raw features don't express.
#
# WHY normalized sum (not PCA or weighted)?
#
#   Option A: PCA (Principal Component Analysis)
#     → Finds the direction of maximum variance — but variance ≠ churn signal.
#     → PCA might weight Total Charges heavily just because it has the widest
#       range, not because it's most relevant to churn.
#     → Also loses interpretability: "PC1 = 0.6" means nothing to a business.
#
#   Option B: Weighted sum with hand-picked weights
#     → Requires assumptions about relative importance before modeling.
#     → If we guess wrong, we inject bias. Let the model learn weights.
#
#   Option C: Min-max normalized sum (our choice)
#     → Normalize each component to [0,1], then average.
#     → Each component contributes equally (no scale bias).
#     → Interpretable: score of 0.8 means customer is in the top 20% across
#       all three engagement dimensions.
#     → Simple and transparent — no hidden assumptions.
#     → The model can still learn to weight components differently by
#       adjusting coefficients on this score vs raw features.
# ============================================================================

print("4.6 — Creating Engagement Score")

tenure_norm = (df['Tenure Months'] - df['Tenure Months'].min()) / \
              (df['Tenure Months'].max() - df['Tenure Months'].min())

charges_norm = (df['Monthly Charges'] - df['Monthly Charges'].min()) / \
               (df['Monthly Charges'].max() - df['Monthly Charges'].min())

addons_norm = df['Num Addon Services'] / 6  # max is 6 add-ons

df['Engagement Score'] = (tenure_norm + charges_norm + addons_norm) / 3

print(f"     Range: {df['Engagement Score'].min():.3f} — {df['Engagement Score'].max():.3f}")
print(f"     Mean:  {df['Engagement Score'].mean():.3f}")
print(f"     Correlation with Churn: {df['Engagement Score'].corr(df['Churn Value']):.3f}")
print()


# ============================================================================
# 4.7 VALIDATE ENGINEERED FEATURES — Do They Actually Help?
# ============================================================================
#
# A feature is only worth keeping if it has predictive power.
# We validate by checking correlation with the target and comparing
# churn rate separation across categories/bins.
# ============================================================================

print("=" * 65)
print(" 4.7 — FEATURE VALIDATION: Do engineered features help?")
print("=" * 65)

# Numerical engineered features: correlation with target
print("\n  Correlation with Churn Value:")
eng_numeric = ['Num Addon Services', 'Avg Monthly Spend', 'Engagement Score']
for col in eng_numeric:
    corr = df[col].corr(df['Churn Value'])
    strength = 'STRONG' if abs(corr) > 0.2 else 'MODERATE' if abs(corr) > 0.1 else 'WEAK'
    print(f"    {col}: r = {corr:.3f} ({strength})")

# Categorical engineered features: churn rate range
print("\n  Churn rate range across categories:")
eng_categorical = ['Tenure Group', 'Charges Tier', 'Contract_Internet']
for col in eng_categorical:
    rates = df.groupby(col, observed=True)['Churn Value'].mean()
    spread = rates.max() - rates.min()
    print(f"    {col}: {rates.min():.1%} — {rates.max():.1%} (spread = {spread:.1%})")


# ============================================================================
# 4.8 VISUALIZATION — Engineered Features vs Churn
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Tenure Group
churn_tenure = df.groupby('Tenure Group', observed=True)['Churn Value'].mean()
colors_t = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.2 else '#2ecc71' for r in churn_tenure.values]
axes[0, 0].bar(range(len(churn_tenure)), churn_tenure.values, color=colors_t, edgecolor='black')
axes[0, 0].set_xticks(range(len(churn_tenure)))
axes[0, 0].set_xticklabels(churn_tenure.index, fontsize=10)
axes[0, 0].set_title('Tenure Group → Churn Rate', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Churn Rate')
for i, v in enumerate(churn_tenure.values):
    axes[0, 0].text(i, v + 0.01, f'{v:.0%}', ha='center', fontweight='bold')

# 2. Charges Tier
churn_charges = df.groupby('Charges Tier', observed=True)['Churn Value'].mean()
colors_c = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.2 else '#2ecc71' for r in churn_charges.values]
axes[0, 1].bar(range(len(churn_charges)), churn_charges.values, color=colors_c, edgecolor='black')
axes[0, 1].set_xticks(range(len(churn_charges)))
axes[0, 1].set_xticklabels(churn_charges.index, fontsize=10)
axes[0, 1].set_title('Charges Tier → Churn Rate', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Churn Rate')
for i, v in enumerate(churn_charges.values):
    axes[0, 1].text(i, v + 0.01, f'{v:.0%}', ha='center', fontweight='bold')

# 3. Num Addon Services
churn_addons = df.groupby('Num Addon Services')['Churn Value'].mean()
colors_a = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.2 else '#2ecc71' for r in churn_addons.values]
axes[0, 2].bar(churn_addons.index, churn_addons.values, color=colors_a, edgecolor='black')
axes[0, 2].set_title('Num Add-on Services → Churn Rate', fontsize=13, fontweight='bold')
axes[0, 2].set_ylabel('Churn Rate')
axes[0, 2].set_xlabel('Number of Add-ons')
for i, v in zip(churn_addons.index, churn_addons.values):
    axes[0, 2].text(i, v + 0.01, f'{v:.0%}', ha='center', fontweight='bold')

# 4. Engagement Score distribution by churn
for val, color, label in [(0, '#2ecc71', 'Stayed'), (1, '#e74c3c', 'Churned')]:
    subset = df[df['Churn Value'] == val]['Engagement Score']
    axes[1, 0].hist(subset, bins=30, alpha=0.5, color=color, density=True, label=label)
    subset.plot.kde(ax=axes[1, 0], color=color, linewidth=2)
axes[1, 0].set_title('Engagement Score by Churn', fontsize=13, fontweight='bold')
axes[1, 0].legend()

# 5. Avg Monthly Spend by churn
for val, color, label in [(0, '#2ecc71', 'Stayed'), (1, '#e74c3c', 'Churned')]:
    subset = df[df['Churn Value'] == val]['Avg Monthly Spend']
    axes[1, 1].hist(subset, bins=30, alpha=0.5, color=color, density=True, label=label)
    subset.plot.kde(ax=axes[1, 1], color=color, linewidth=2)
axes[1, 1].set_title('Avg Monthly Spend by Churn', fontsize=13, fontweight='bold')
axes[1, 1].legend()

# 6. Contract_Internet top combinations
top_combos = df.groupby('Contract_Internet')['Churn Value'].mean().sort_values(ascending=False).head(6)
colors_ci = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.2 else '#2ecc71' for r in top_combos.values]
axes[1, 2].barh(range(len(top_combos)), top_combos.values, color=colors_ci, edgecolor='black')
axes[1, 2].set_yticks(range(len(top_combos)))
axes[1, 2].set_yticklabels(top_combos.index, fontsize=9)
axes[1, 2].set_title('Contract × Internet → Churn Rate (Top 6)', fontsize=13, fontweight='bold')
axes[1, 2].set_xlabel('Churn Rate')
for i, v in enumerate(top_combos.values):
    axes[1, 2].text(v + 0.01, i, f'{v:.0%}', va='center', fontweight='bold')

plt.suptitle('Engineered Features — Validation Against Churn',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'plot_12_engineered_features.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# 4.9 SAVE ENGINEERED DATA
# ============================================================================

print(f"\n  Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  New columns added: Tenure Group, Charges Tier, Num Addon Services,")
print(f"                     Contract_Internet, Avg Monthly Spend, Engagement Score")
df.to_csv(os.path.join(DATA_PROCESSED, 'telco_churn_engineered.csv'), index=False)
print(f"  Saved: data/processed/telco_churn_engineered.csv\n")


# ============================================================================
# STEP 4 SUMMARY
# ============================================================================

print("=" * 65)
print(" STEP 4 — FEATURE ENGINEERING SUMMARY")
print("=" * 65)
print("""
 Feature Created         | What It Captures           | Why This Approach
 ------------------------|----------------------------|----------------------------
 Tenure Group            | Customer lifecycle stage   | Domain-driven bins match
  (4 bins)               | (new → loyal)              | churn breakpoints from EDA
                         |                            |
 Num Addon Services      | Product penetration depth  | Replaces 6 correlated cols
  (count 0-6)            | (stickiness)               | with 1 informative count
                         |                            |
 Charges Tier            | Pricing segment            | Bimodal distribution has
  (Low/Mid/High)         | (basic → premium)          | natural tier boundaries
                         |                            |
 Contract_Internet       | Risk combination           | 51% churn for fiber+MTM
  (interaction)          | (contract × service type)  | vs 1% for no-internet+2yr
                         |                            |
 Avg Monthly Spend       | True average billing rate  | Differs from current rate
  (Total / Tenure)       | (includes plan changes)    | if customer upgraded/downgraded
                         |                            |
 Engagement Score        | Composite loyalty metric   | Min-max normalized sum of
  (0-1 composite)        | (tenure + spend + add-ons) | tenure, charges, add-ons

 NEXT: Step 5 — Feature Selection / Encoding
""")
print("=" * 65)
