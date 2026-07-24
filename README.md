# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn for a telecom company using Gradient Boosting, with SHAP-based explainability and an interactive Streamlit dashboard.

## Live Application

**[Launch the Dashboard](https://mohitm1530-telco-customer-churn.streamlit.app)**

---

## Problem Statement

Customer churn is one of the most critical metrics for telecom companies. Acquiring a new customer costs **5-7x more** than retaining an existing one. This project builds a predictive model that identifies at-risk customers before they leave, enabling proactive retention strategies.

## Key Results

| Metric | Value |
|:--|:--|
| ROC-AUC | 0.856 |
| PR-AUC | 0.674 |
| Recall (Churners) | 88.5% |
| F2 Score | 0.757 |
| Decision Threshold | 0.160 (optimized for recall) |

The model catches **88.5% of actual churners** at the cost of some false positives -- the right trade-off when a retention call is cheap but losing a customer is expensive.

## Dashboard

The Streamlit app has four tabs:

- **Predict** -- Enter a single customer's details and get a churn probability with SHAP-based explanation of the top drivers
- **Batch Score** -- Upload a CSV of customers, get scored predictions with risk levels, and download the results
- **Feature Insights** -- Business-friendly interpretation of what drives churn, with actionable recommendations
- **Model Performance** -- Test-set metrics, confusion matrix, ROC and PR curves with plain-English explanations

## ML Pipeline

The project follows a structured 8-step pipeline, each implemented as a standalone script in `src/`:

| Step | Script | Description |
|:--:|:--|:--|
| 1 | `01_eda.py` | Exploratory data analysis with 10 diagnostic plots |
| 2 | `02_data_cleaning.py` | Handle missing values, fix data types, remove duplicates |
| 3 | `03_feature_engineering.py` | Create tenure groups, charges tiers, contract-internet interactions, addon counts |
| 4 | `04_feature_selection.py` | Remove multicollinear features, keep 38 final predictors |
| 5 | `05_data_splitting.py` | Stratified 80/20 train-test split with StandardScaler on numerical columns |
| 6 | `06_class_imbalance.py` | SMOTE oversampling to handle 73.5%/26.5% class imbalance |
| 7 | `07_leakage_checks.py` | Verify no data leakage between train and test sets |
| 8 | `08_modelling_and_shap.py` | Train 5 models, Optuna hyperparameter tuning, SHAP explainability |

## Top Churn Drivers (SHAP Analysis)

1. **Customer Tenure** -- Customers under 12 months are 3x more likely to churn
2. **Month-to-Month + Fiber Optic** -- 51% churn rate, the highest of any segment
3. **No Dependents** -- Single customers churn at significantly higher rates
4. **Two-Year Contract** -- Strongest churn shield, rarely leave even when dissatisfied
5. **Electronic Check Payment** -- 2x churn rate vs. auto-pay users

## Project Structure

```
telco-customer-churn/
|-- app.py                     # Streamlit dashboard
|-- requirements.txt           # Python dependencies
|-- .streamlit/config.toml     # Streamlit theme configuration
|-- src/                       # ML pipeline scripts (01-08)
|-- artifacts/                 # Trained model & scaler (.joblib)
|-- data/
|   |-- raw/                   # Original dataset
|   |-- processed/             # Cleaned & engineered datasets
|   +-- splits/                # Train/test splits
+-- reports/                   # Generated plots & SHAP analysis
```

## Tech Stack

- **ML**: scikit-learn, XGBoost, Optuna, SHAP, imbalanced-learn
- **Dashboard**: Streamlit, Plotly
- **Data**: pandas, NumPy, SciPy

## Run Locally

```bash
# Clone the repo
git clone https://github.com/mohitm1530/telco-customer-churn.git
cd telco-customer-churn

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) -- 7,043 customers with 21 attributes including demographics, account info, and service subscriptions.

## License

This project is open source and available under the [MIT License](LICENSE).
