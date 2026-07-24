"""
Telco Customer Churn — Prediction Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import shap
import os
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'artifacts')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')

# ── Palette ──
PAL = {
    "primary": "#4F46E5",
    "primary_light": "#EEF2FF",
    "danger": "#DC2626",
    "danger_light": "#FEF2F2",
    "success": "#059669",
    "success_light": "#ECFDF5",
    "warning": "#D97706",
    "warning_light": "#FFFBEB",
    "text": "#0F172A",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "bg": "#FFFFFF",
    "bg_alt": "#F8FAFC",
}

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# GLOBAL CSS
# ============================================================================

st.markdown("""
<style>
    /* ── Reset & Base ── */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1200px; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Section titles ── */
    .sec-title {
        font-size: 1.15rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: inherit;
        margin: 1.8rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4F46E5;
    }

    /* ── KPI row ── */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .kpi-val { font-size: 1.65rem; font-weight: 700; line-height: 1.3; }
    .kpi-lbl { font-size: 0.72rem; font-weight: 600; color: #64748B;
               text-transform: uppercase; letter-spacing: 0.6px; margin-top: 0.15rem; }

    /* ── Prediction result card ── */
    .result-card {
        border-radius: 8px;
        padding: 1.6rem;
        text-align: center;
    }
    .result-pct  { font-size: 2.8rem; font-weight: 800; line-height: 1.1; }
    .result-tag  { display: inline-block; padding: 0.25rem 0.8rem; border-radius: 4px;
                   font-size: 0.72rem; font-weight: 700; letter-spacing: 0.8px;
                   margin-top: 0.6rem; }
    .result-sub  { font-size: 0.78rem; color: #64748B; margin-top: 0.6rem; }

    /* ── Driver chips ── */
    .driver-chip {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem 0.15rem;
    }
    .chip-up   { background: #FEF2F2; color: #DC2626; }
    .chip-down { background: #ECFDF5; color: #059669; }

    /* ── Hide sidebar toggle ── */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

    /* ── Tab text ── */
    button[data-baseweb="tab"] > div > p { font-size: 0.85rem; font-weight: 600; }

    /* ── Table override ── */
    .stDataFrame { border-radius: 6px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

import sys
try:
    import sklearn._loss
    sys.modules.setdefault('_loss', sklearn._loss)
except ImportError:
    pass
try:
    import sklearn._loss.loss
    sys.modules.setdefault('_loss.loss', sklearn._loss.loss)
except ImportError:
    pass
try:
    import sklearn._loss.link
    sys.modules.setdefault('_loss.link', sklearn._loss.link)
except ImportError:
    pass


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(os.path.join(ARTIFACTS_DIR, 'best_churn_model.joblib'))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
    return bundle['model'], bundle['threshold'], bundle, scaler

model, threshold, model_info, scaler = load_artifacts()
FEATURES = model_info['features']
SCALED_COLS = ['Tenure Months', 'Monthly Charges', 'Num Addon Services',
               'Tenure Group', 'Charges Tier']


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def _encode_service(val):
    return {'No internet service': (1, 0), 'Yes': (0, 1), 'No': (0, 0)}.get(val, (0, 0))


def build_feature_vector(inp: dict) -> pd.DataFrame:
    tenure = inp['tenure']
    monthly = inp['monthly_charges']
    contract = inp['contract']
    internet = inp['internet_service']
    payment = inp['payment_method']

    # Ordinal
    tg = 3
    for i, edge in enumerate([12, 24, 48, 72]):
        if tenure <= edge:
            tg = i; break
    ct = 2
    for i, edge in enumerate([35, 70, 120]):
        if monthly <= edge:
            ct = i; break

    addons = ['online_security', 'online_backup', 'device_protection',
              'tech_support', 'streaming_tv', 'streaming_movies']
    n_addons = sum(1 for a in addons if inp[a] == 'Yes')

    ci = f"{contract}_{internet}"
    ci_keys = ['Month-to-month_Fiber optic', 'Month-to-month_No',
               'One year_DSL', 'One year_Fiber optic', 'One year_No',
               'Two year_DSL', 'Two year_Fiber optic', 'Two year_No']

    os_nis, os_yes = _encode_service(inp['online_security'])
    ts_nis, ts_yes = _encode_service(inp['tech_support'])
    ob_nis, ob_yes = _encode_service(inp['online_backup'])
    dp_nis, dp_yes = _encode_service(inp['device_protection'])
    sm_nis, sm_yes = _encode_service(inp['streaming_movies'])
    sv_nis, sv_yes = _encode_service(inp['streaming_tv'])

    row = {
        'Tenure Months': tenure, 'Tenure Group': tg, 'Monthly Charges': monthly,
        'Dependents': int(inp['dependents'] == 'Yes'),
        'Charges Tier': ct, 'Num Addon Services': n_addons,
        'Paperless Billing': int(inp['paperless_billing'] == 'Yes'),
        'Partner': int(inp['partner'] == 'Yes'),
        'Senior Citizen': int(inp['senior_citizen'] == 'Yes'),
        **{f'Contract_Internet_{k}': int(ci == k) for k in ci_keys},
        'Contract_One year': int(contract == 'One year'),
        'Contract_Two year': int(contract == 'Two year'),
        'Online Security_No internet service': os_nis, 'Online Security_Yes': os_yes,
        'Tech Support_No internet service': ts_nis, 'Tech Support_Yes': ts_yes,
        'Internet Service_Fiber optic': int(internet == 'Fiber optic'),
        'Internet Service_No': int(internet == 'No'),
        'Online Backup_No internet service': ob_nis, 'Online Backup_Yes': ob_yes,
        'Payment Method_Credit card (automatic)': int(payment == 'Credit card (automatic)'),
        'Payment Method_Electronic check': int(payment == 'Electronic check'),
        'Payment Method_Mailed check': int(payment == 'Mailed check'),
        'Device Protection_No internet service': dp_nis, 'Device Protection_Yes': dp_yes,
        'Streaming Movies_No internet service': sm_nis, 'Streaming Movies_Yes': sm_yes,
        'Streaming TV_No internet service': sv_nis, 'Streaming TV_Yes': sv_yes,
        'Multiple Lines_No phone service': int(inp['multiple_lines'] == 'No phone service'),
        'Multiple Lines_Yes': int(inp['multiple_lines'] == 'Yes'),
    }
    df = pd.DataFrame([row])[FEATURES]
    df[SCALED_COLS] = scaler.transform(df[SCALED_COLS])
    return df


def build_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_raw.iterrows():
        inp = {
            'tenure': r.get('Tenure Months', r.get('tenure', 0)),
            'monthly_charges': r.get('Monthly Charges', r.get('monthly_charges', 0)),
            'contract': r.get('Contract', 'Month-to-month'),
            'internet_service': r.get('Internet Service', 'No'),
            'payment_method': r.get('Payment Method', 'Bank transfer (automatic)'),
            'dependents': r.get('Dependents', 'No'),
            'partner': r.get('Partner', 'No'),
            'senior_citizen': r.get('Senior Citizen', 'No'),
            'paperless_billing': r.get('Paperless Billing', 'No'),
            'phone_service': r.get('Phone Service', 'Yes'),
            'multiple_lines': r.get('Multiple Lines', 'No'),
            'online_security': r.get('Online Security', 'No'),
            'online_backup': r.get('Online Backup', 'No'),
            'device_protection': r.get('Device Protection', 'No'),
            'tech_support': r.get('Tech Support', 'No'),
            'streaming_tv': r.get('Streaming TV', 'No'),
            'streaming_movies': r.get('Streaming Movies', 'No'),
        }
        rows.append(build_feature_vector(inp))
    return pd.concat(rows, ignore_index=True)


# ============================================================================
# PLOTLY DEFAULTS
# ============================================================================

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=PAL["text"]),
    margin=dict(l=50, r=20, t=44, b=50),
    hoverlabel=dict(font_size=12),
)


def _layout(**overrides):
    """Merge _PLOTLY_LAYOUT with per-chart overrides (overrides win)."""
    merged = {**_PLOTLY_LAYOUT, **overrides}
    return merged


def _base_fig(**kw):
    fig = go.Figure()
    fig.update_layout(**_PLOTLY_LAYOUT, **kw)
    return fig


# ============================================================================
# MAIN AREA — TABS
# ============================================================================

tab1, tab2, tab4, tab3 = st.tabs([
    "Predict", "Batch Score", "Feature Insights", "Model Performance",
])


# ============================  TAB 1  =======================================

with tab1:
    st.markdown('<p class="sec-title">Customer Information</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check"])

    with c2:
        st.markdown("**Services**")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        if internet == "No":
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
            st.info("Internet service is required for add-ons.")
        else:
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    with c3:
        st.markdown("**Demographics & Billing**")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
        if phone_svc == "No":
            multiple_lines = "No phone service"
        else:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

    st.markdown("")

    if st.button("Run Prediction", type="primary", use_container_width=True):
        inp = dict(tenure=tenure, monthly_charges=monthly, contract=contract,
                   internet_service=internet, payment_method=payment,
                   dependents=dependents, partner=partner, senior_citizen=senior,
                   paperless_billing=paperless, phone_service=phone_svc,
                   multiple_lines=multiple_lines, online_security=online_security,
                   online_backup=online_backup, device_protection=device_protection,
                   tech_support=tech_support, streaming_tv=streaming_tv,
                   streaming_movies=streaming_movies)

        X = build_feature_vector(inp)
        prob = model.predict_proba(X)[0][1]
        pred = int(prob >= threshold)

        if prob < 0.20:
            tag, color, bg = "LOW RISK", PAL["success"], PAL["success_light"]
        elif prob < 0.40:
            tag, color, bg = "MODERATE", PAL["warning"], PAL["warning_light"]
        elif prob < 0.70:
            tag, color, bg = "HIGH RISK", PAL["danger"], "#FEE2E2"
        else:
            tag, color, bg = "CRITICAL", PAL["danger"], "#FECACA"

        st.markdown('<p class="sec-title">Prediction Result</p>', unsafe_allow_html=True)

        rc, gc = st.columns([1, 2], gap="medium")

        with rc:
            st.markdown(f"""
            <div class="result-card" style="background:{bg}; border:1px solid {color}22;">
                <div class="result-pct" style="color:{color}">{prob:.0%}</div>
                <div class="result-tag" style="background:{color}; color:#fff;">{tag}</div>
                <div class="result-sub">
                    Verdict: <b>{'Will Churn' if pred else 'Will Stay'}</b><br>
                    Decision threshold: {threshold:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with gc:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            shap_s = pd.Series(sv[0], index=FEATURES)
            top = shap_s.abs().sort_values(ascending=False).head(8)
            names = [f.replace('_', ' ') for f in top.index]
            vals = [shap_s[f] for f in top.index]
            colors = [PAL["danger"] if v > 0 else PAL["success"] for v in vals]

            fig = go.Figure(go.Bar(
                x=vals[::-1], y=names[::-1], orientation='h',
                marker_color=colors[::-1],
                hovertemplate='%{y}: %{x:.3f}<extra></extra>',
            ))
            fig.update_layout(
                **_layout(margin=dict(l=20, r=20, t=40, b=40)),
                title=dict(text="Why This Prediction?", font_size=14),
                height=340,
                xaxis=dict(title="Impact on prediction", zeroline=True,
                           zerolinecolor=PAL["border"], gridcolor="#F1F5F9"),
                yaxis=dict(tickfont_size=12),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Driver chips
        chips = ""
        for feat in top.index[:6]:
            v = shap_s[feat]
            cls = "chip-up" if v > 0 else "chip-down"
            arrow = "↑" if v > 0 else "↓"
            chips += f'<span class="driver-chip {cls}">{arrow} {feat.replace("_"," ")}</span>'
        st.markdown(chips, unsafe_allow_html=True)


# ============================  TAB 2  =======================================

with tab2:
    st.markdown('<p class="sec-title">Upload Customer Data</p>', unsafe_allow_html=True)

    st.markdown(
        "Upload a CSV with columns: `Tenure Months`, `Monthly Charges`, `Contract`, "
        "`Internet Service`, `Payment Method`, `Dependents`, `Partner`, `Senior Citizen`, "
        "`Paperless Billing`, `Phone Service`, `Multiple Lines`, `Online Security`, "
        "`Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`"
    )

    uploaded = st.file_uploader("Choose CSV", type="csv", label_visibility="collapsed")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        with st.spinner("Scoring..."):
            X_b = build_batch(df_up)
            probs = model.predict_proba(X_b)[:, 1]
            preds = (probs >= threshold).astype(int)

        df_up['Churn Probability'] = probs
        df_up['Prediction'] = np.where(preds, 'Churn', 'Stay')
        df_up['Risk Level'] = pd.cut(probs, [0, 0.2, 0.4, 0.7, 1.0],
                                     labels=['Low', 'Moderate', 'High', 'Critical'])

        # KPIs
        st.markdown('<p class="sec-title">Summary</p>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4, gap="medium")
        items = [
            (k1, preds.sum(), "Predicted Churners", PAL["danger"]),
            (k2, (preds == 0).sum(), "Predicted Staying", PAL["success"]),
            (k3, f"{probs.mean():.1%}", "Avg Churn Prob", PAL["primary"]),
            (k4, (probs >= 0.7).sum(), "Critical Risk", PAL["warning"]),
        ]
        for col, val, lbl, clr in items:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-val" style="color:{clr}">{val}</div>
                    <div class="kpi-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<p class="sec-title">Distribution & Details</p>', unsafe_allow_html=True)
        dc, tc = st.columns([1, 2], gap="medium")

        with dc:
            risk_counts = df_up['Risk Level'].value_counts()
            fig = go.Figure(go.Pie(
                labels=risk_counts.index, values=risk_counts.values,
                hole=0.55, sort=False,
                marker_colors=[PAL["success"], PAL["warning"], "#F97316", PAL["danger"]],
                textinfo='label+value', textposition='outside',
                hovertemplate='%{label}: %{value} customers<extra></extra>',
            ))
            fig.update_layout(**_PLOTLY_LAYOUT, height=320, showlegend=False,
                              title=dict(text="Risk Breakdown", font_size=14))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with tc:
            show_cols = [c for c in ['Tenure Months', 'Contract', 'Internet Service',
                                     'Monthly Charges'] if c in df_up.columns]
            show_cols += ['Churn Probability', 'Prediction', 'Risk Level']
            st.dataframe(
                df_up[show_cols].sort_values('Churn Probability', ascending=False).head(20),
                use_container_width=True, hide_index=True,
            )

        csv_out = df_up.to_csv(index=False)
        st.download_button("Download Scored CSV", csv_out,
                           "churn_predictions.csv", "text/csv",
                           use_container_width=True)


# ============================  TAB 3  =======================================

with tab3:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, fbeta_score, roc_auc_score,
                                 average_precision_score, confusion_matrix,
                                 roc_curve, precision_recall_curve)

    X_test = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'splits', 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'splits', 'y_test.csv')).squeeze()
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    f2   = fbeta_score(y_test, y_pred, beta=2)
    roc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)

    # KPI row
    st.markdown('<p class="sec-title">Test-Set Metrics</p>', unsafe_allow_html=True)
    cols = st.columns(7, gap="small")
    for col, (lbl, val, clr) in zip(cols, [
        ("Accuracy", acc, PAL["muted"]), ("Precision", prec, PAL["primary"]),
        ("Recall", rec, PAL["success"]), ("F1", f1, PAL["warning"]),
        ("F2", f2, PAL["danger"]), ("ROC-AUC", roc, PAL["primary"]),
        ("PR-AUC", ap, PAL["danger"]),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val" style="color:{clr}">{val:.3f}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # Charts — confusion matrix + curves
    st.markdown('<p class="sec-title">Diagnostic Charts</p>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        cm = confusion_matrix(y_test, y_pred)
        labels = ['Stay', 'Churn']
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#EEF2FF"], [1, PAL["primary"]]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=22, color="white"),
            hovertemplate='Actual %{y} → Predicted %{x}: %{z}<extra></extra>',
            showscale=False,
        ))
        _cm_margin = dict(l=60, r=30, t=50, b=60)
        fig.update_layout(**_layout(margin=_cm_margin), height=400,
                          title=dict(text="Confusion Matrix", font_size=14),
                          xaxis=dict(title="Predicted", side="bottom"),
                          yaxis=dict(title="Actual", autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 line=dict(color=PAL["primary"], width=2.5),
                                 fill='tozeroy', fillcolor="rgba(79,70,229,0.08)",
                                 name=f'ROC-AUC = {roc:.3f}',
                                 hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 line=dict(color=PAL["border"], dash='dash', width=1),
                                 showlegend=False, hoverinfo='skip'))
        fig.update_layout(**_layout(margin=_cm_margin), height=400,
                          title=dict(text="ROC Curve", font_size=14),
                          xaxis=dict(title="False Positive Rate", gridcolor="#F1F5F9"),
                          yaxis=dict(title="True Positive Rate", gridcolor="#F1F5F9"),
                          legend=dict(x=0.55, y=0.08, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    pr1, pr2 = st.columns(2, gap="large")

    with pr1:
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec_c, y=prec_c, mode='lines',
                                 line=dict(color=PAL["danger"], width=2.5),
                                 fill='tozeroy', fillcolor="rgba(220,38,38,0.08)",
                                 name=f'PR-AUC = {ap:.3f}',
                                 hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'))
        fig.update_layout(**_layout(margin=_cm_margin), height=400,
                          title=dict(text="Precision-Recall Curve", font_size=14),
                          xaxis=dict(title="Recall", gridcolor="#F1F5F9"),
                          yaxis=dict(title="Precision", gridcolor="#F1F5F9"),
                          legend=dict(x=0.55, y=0.95, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with pr2:
        st.markdown("")
        st.markdown("""
**What these charts tell us**

**Confusion Matrix** — Of 1,409 test customers, the model correctly
identified 331 out of 374 actual churners (88.5 % recall). The trade-off:
360 false alarms among loyal customers — acceptable when a retention
call costs far less than losing a customer.

**ROC Curve** — AUC of 0.856 means the model distinguishes churners
from loyal customers with strong accuracy across all thresholds.

**PR Curve** — AP of 0.673 reflects performance on the minority class
(churners are only 26.5 % of the data). This is the more honest metric
for imbalanced datasets.
""")

    # Model details
    st.markdown('<p class="sec-title">Model Details</p>', unsafe_allow_html=True)
    d1, d2 = st.columns([1, 1], gap="medium")
    with d1:
        st.markdown(f"""
| Property | Value |
|:--|:--|
| Algorithm | {model_info['model_name']} |
| Decision Threshold | {threshold:.3f} |
| PR-AUC (test) | {model_info['test_prauc']:.4f} |
| ROC-AUC (test) | {model_info['test_roc_auc']:.4f} |
| F2 Score (test) | {model_info['test_f2']:.4f} |
| Features | {len(FEATURES)} |
| Train Samples | 5,634 |
| Test Samples | 1,409 |
""")
    with d2:
        st.markdown("""
**Why F2 over F1?**

Missing a churner is more costly than a false alarm.
F2 weights recall 4x more than precision.
Our optimized threshold (0.160) catches **88.5 % of churners**
at the cost of some false positives — the right trade-off
when a retention call is cheap but losing a customer is expensive.
""")


# ============================  TAB 4  =======================================

with tab4:
    fi_df = pd.read_csv(os.path.join(REPORTS_DIR, 'shap', 'shap_feature_importance.csv'))

    # ── Business Interpretation (TOP) ──
    st.markdown('<p class="sec-title">Key Findings &amp; Business Actions</p>',
                unsafe_allow_html=True)

    st.markdown("""
The model identifies which customer attributes most strongly predict churn.
The table below translates those findings into actionable recommendations.
""")

    st.markdown("""
| Rank | Driver | What the Data Shows | Business Impact | Recommended Action |
|:--:|:--|:--|:--|:--|
| 1 | **Customer Tenure** | Customers under 12 months are **3x more likely** to churn than those above 48 months. Every additional month reduces churn risk. | New customers are the most vulnerable — early churn destroys acquisition ROI. | Launch a 90-day onboarding program. Assign dedicated reps for the first year. Offer loyalty milestones (e.g., 6-month discount). |
| 2 | **Month-to-Month + Fiber Optic** | This combination has a **51 % churn rate** — the highest of any segment. No lock-in plus premium pricing creates easy exit conditions. | This segment generates the bulk of voluntary churn and is the largest revenue leak. | Proactively offer 12-month upgrades with a price-lock guarantee. Bundle add-on services at a discount to increase switching cost. |
| 3 | **No Dependents** | Single customers without dependents churn at significantly higher rates. Family accounts create natural stickiness through shared plans. | Solo customers have fewer ties to the service, making them price-sensitive and easier to lose. | Design engagement programs for singles — content bundles, referral bonuses, or community perks that build non-price loyalty. |
| 4 | **Two-Year Contract** | Two-year contracts are the strongest churn shield. These customers rarely leave, even when dissatisfied. | Long-term contracts stabilize revenue and reduce churn management overhead. | Incentivize upgrades — waived installation, free premium channels for 3 months, or a bill credit upon signing. |
| 5 | **Electronic Check** | Customers paying by electronic check churn at nearly **2x the rate** of auto-pay users. Manual payment creates a monthly "should I keep this?" moment. | Each manual payment is a decision point that auto-pay eliminates entirely. | Offer a recurring $5/month credit for auto-pay enrollment. Highlight savings during onboarding. |
""")

    # ── Feature Importance Chart ──
    st.markdown('<p class="sec-title">What Drives Churn the Most?</p>',
                unsafe_allow_html=True)

    top_n = 10
    plot_df = fi_df.head(top_n).iloc[::-1].copy()
    plot_df['Label'] = plot_df['Feature'].str.replace('_', ' ')

    CLR_RISK = "#FF6B6B"
    CLR_SAFE = "#4ECDC4"

    protect_set = {'Tenure Months', 'Contract_Two year',
                   'Online Security_Yes', 'Tech Support_Yes'}

    bar_colors = [CLR_SAFE if row['Feature'] in protect_set else CLR_RISK
                  for _, row in plot_df.iterrows()]

    fig = go.Figure(go.Bar(
        x=plot_df['Mean_SHAP'], y=plot_df['Label'], orientation='h',
        marker_color=bar_colors, showlegend=False,
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=CLR_RISK,
                         name="Increases Churn Risk", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=CLR_SAFE,
                         name="Protects Against Churn", showlegend=True))
    fig.update_layout(
        **_layout(margin=dict(l=20, r=20, t=70, b=50)),
        height=450,
        title=dict(text="Top 10 Features by Influence on Churn", font_size=14),
        xaxis=dict(title="Average Impact on Prediction",
                   gridcolor="rgba(255,255,255,0.08)", zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(tickfont_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center",
                    x=0.5, font_size=13),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
**How to read this chart** — Longer bars mean more influence on the prediction.
Warm-toned bars are factors that increase churn risk (e.g., month-to-month contracts, electronic check).
Cool-toned bars are protective factors that reduce churn risk (e.g., long tenure, two-year contracts).
The top 3 drivers alone account for over 70 % of the model's decision-making.
""")
