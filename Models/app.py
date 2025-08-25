# app.py
# Streamlit demo for Healthcare Fraud Detection (NHIS Healthcare Claims & Fraud)
# Kaggle dataset: https://www.kaggle.com/datasets/bonifacechosen/nhis-healthcare-claims-and-fraud-dataset

import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional imports for model loading and explainability
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

# SHAP is optional; code will still run without it
try:
    import shap  # pip install shap
except Exception:  # pragma: no cover
    shap = None


# =============================
# THEME (Daewoong vibes)
# =============================
PRIMARY = "#F26B21"     # Daewoong orange
PRIMARY_DARK = "#C45319"
BG = "#0F1117"
CARD = "#1B1F2A"
TEXT = "#E6E6E6"

st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🧾",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background:{BG}; color:{TEXT}; }}
    section[data-testid="stSidebar"] {{
        background:#161A23; border-right:1px solid #232838;
    }}
    .daewoong-hero {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        color: white; padding: 28px 36px; border-radius: 18px; text-align: center;
        box-shadow: 0 10px 24px rgba(0,0,0,.45); margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,.08);
    }}
    .page-title {{ font-size: 2.6rem; font-weight: 800; margin: 2px 0 2px 0; }}
    .card {{
        background: {CARD}; border: 1px solid #262C3C; border-radius: 16px;
        padding: 18px 20px; box-shadow: 0 6px 18px rgba(0,0,0,.35);
    }}
    .stButton > button {{
        background: {PRIMARY}; color: white; border-radius: 12px; border: 1px solid {PRIMARY_DARK};
        padding: 8px 16px; font-weight: 600;
    }}
    .stButton > button:hover {{ background: {PRIMARY_DARK}; border-color: {PRIMARY_DARK}; }}
    div[data-baseweb="select"] > div:focus-within {{ outline: 2px solid {PRIMARY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# UTILITIES
# =============================
def hero(title="Healthcare Fraud Detection App", subtitle="Made for: SIU / Audit Team"):
    st.markdown(
        f"""
        <div class="daewoong-hero">
            <div class="page-title">{title}</div>
            <div style="opacity:.9; letter-spacing:.3px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def find_first(paths):
    """Return first existing Path from a list of filenames."""
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load models, preprocessor, scaler, optional metrics/leaderboard."""
    models = {}
    model_files = {
        "XGBoost": "model_xgb.pkl",
        "LightGBM": "model_lgbm.pkl",
        "Random Forest": "model_rf.pkl",
        "Logistic Regression": "model_logreg.pkl",
        "CatBoost": "model_cat.pkl",
        "Model": "model.pkl",
    }

    if joblib is not None:
        for name, fname in model_files.items():
            if Path(fname).exists():
                try:
                    models[name] = joblib.load(fname)
                except Exception as e:
                    st.warning(f"Failed to load {fname}: {e}")

    preprocessor = joblib.load("preprocessor.pkl") if (joblib and Path("preprocessor.pkl").exists()) else None
    scaler = joblib.load("scaler.pkl") if (joblib and Path("scaler.pkl").exists()) else None

    comp_df = pd.read_csv("model_comparison.csv") if Path("model_comparison.csv").exists() else None
    metrics = json.loads(Path("metrics.json").read_text()) if Path("metrics.json").exists() else None

    return models, preprocessor, scaler, comp_df, metrics

def preprocess_df(df, preprocessor=None, scaler=None, model=None):
    """Apply your saved preprocessing; otherwise fallback to pandas.get_dummies."""
    X = df.copy()

    if preprocessor is not None:
        X = preprocessor.transform(X)  # must be fitted during training

    if scaler is not None:
        # If scaler was applied inside preprocessor, skip; otherwise try to scale numeric cols only.
        try:
            num_cols = X.select_dtypes(include=[np.number]).columns
            X[num_cols] = scaler.transform(X[num_cols])
        except Exception:
            pass

    # Align to model's expected features if available
    if model is not None and hasattr(model, "feature_names_in_"):
        X = pd.DataFrame(X)
        X = X.reindex(columns=list(model.feature_names_in_), fill_value=0)

    # Fallback: one-hot everything and hope columns align
    if model is not None and not hasattr(model, "feature_names_in_") and isinstance(X, pd.DataFrame):
        X = pd.get_dummies(X)

    return X

def predict_proba(model, X):
    """Get probability of positive class (Fraud)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Some linear models expose decision_function; we can squash via sigmoid (approx)
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1 / (1 + np.exp(-z))
    # As last resort, use hard predictions
    y = model.predict(X)
    return y.astype(float)

def download_button_from_df(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download scored CSV", data=csv, file_name=filename, mime="text/csv")

def show_shap_explanation(model, X_row):
    """Display local explanation if SHAP + supported model are available."""
    if shap is None:
        st.info("Install SHAP (`pip install shap`) for local explanations.")
        return
    try:
        # Best effort: use TreeExplainer for tree models, otherwise KernelExplainer (slow)
        if model.__class__.__name__.lower() in ("xgbclassifier", "lgbmclassifier", "catboostclassifier", "randomforestclassifier", "gradientboostingclassifier"):
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X_row)
        else:
            # KernelExplainer can be slow; we do a tiny sampling
            f = lambda X: predict_proba(model, X)
            explainer = shap.KernelExplainer(f, X_row)
            vals = explainer.shap_values(X_row, nsamples=100)
        shap.initjs()
        st.subheader("Local explanation (SHAP)")
        try:
            shap.force_plot(explainer.expected_value, vals[0] if isinstance(vals, list) else vals, X_row, matplotlib=True, show=False)
            st.pyplot(bbox_inches="tight")
        except Exception:
            # Fallback to bar chart
            arr = vals[0] if isinstance(vals, list) else vals
            contrib = pd.Series(arr.flatten(), index=getattr(X_row, "columns", range(X_row.shape[1])))
            st.bar_chart(contrib.sort_values(ascending=False).head(10))
    except Exception as e:
        st.info(f"SHAP explanation not available: {e}")

# =============================
# SIDEBAR NAV
# =============================
st.sidebar.markdown("#### Menu")
page = st.sidebar.selectbox(
    "",
    options=["Home", "Machine Learning App"],
    index=0,
    label_visibility="collapsed",
)

# Load assets once
models, preprocessor, scaler, comp_df, metrics = load_artifacts()

# =============================
# HOME
# =============================
if page == "Home":
    hero()

    st.header("Overview")
    st.markdown(
        """
        <div class="card">
        <p><b>Objective:</b> Binary classification to flag likely <b>Fraud</b> vs <b>Non-Fraud</b> healthcare claims so SIU teams can prioritize investigation.</p>
        <p><b>Dataset:</b> NHIS Healthcare Claims & Fraud (Kaggle: <i>bonifacechosen/nhis-healthcare-claims-and-fraud-dataset</i>).</p>
        <p><b>Why it matters:</b> Fraud datasets are imbalanced—missing a fraud (false negative) can be costly, so we expose a <b>threshold slider</b> to trade precision vs. recall.</p>
        <ul>
            <li>Pipeline: Cleaning → Encoding (Target/One-Hot) → Scaling (if used) → Model.</li>
            <li>Models supported: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost (load any you have).</li>
            <li>Outputs: label + probability, local explanation (optional SHAP), and batch CSV scoring.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if comp_df is not None and not comp_df.empty:
        st.subheader("Model Leaderboard (from training)")
        st.caption("Tip: Provide `model_comparison.csv` with columns like: model, roc_auc, recall, precision, f1, fit_time.")
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Place `model_comparison.csv` to show your training leaderboard here.")

    if metrics is not None:
        st.subheader("Hold-out Metrics (static)")
        st.json(metrics)
    else:
        st.caption("Optional: provide `metrics.json` (e.g., {'roc_auc': 0.92, 'recall@0.35': 0.84, ...}).")

    st.markdown(
        f"""
        <div style="margin-top:18px; text-align:center; opacity:.65;">
            Built with Streamlit • Themed in <b style="color:{PRIMARY}">Daewoong Orange</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================
# MACHINE LEARNING APP (SCORING)
# =============================
else:
    hero("Healthcare Fraud Detection — Inference", "Real-time & batch scoring")

    if not models:
        st.error("No model file found. Add `model.pkl` or one of: model_xgb.pkl, model_lgbm.pkl, model_rf.pkl, model_logreg.pkl, model_cat.pkl.")
        st.stop()

    model_name = st.selectbox("Choose model", list(models.keys()))
    model = models[model_name]

    st.markdown("### Single-case Scoring")

    # Inputs (align these to your training features)
    c1, c2 = st.columns(2)
    with c1:
        AGE = st.number_input("AGE", min_value=0, max_value=120, value=45, step=1)
    with c2:
        GENDER = st.selectbox("GENDER", ["M", "F"])

    with c1:
        DIAGNOSIS = st.text_input("DIAGNOSIS", value="Refractive Error")
    with c2:
        ENCOUNTER_DAY = st.selectbox("ENCOUNTER_DAY", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    with c1:
        LENGTH_OF_STAY = st.number_input("LENGTH_OF_STAY (days)", min_value=0, max_value=365, value=2, step=1)
    with c2:
        AMOUNT_BILLED = st.number_input("Amount Billed", min_value=0.0, value=1500.0, step=50.0, format="%.2f")

    thr_col, _ = st.columns([1.3, 3])
    with thr_col:
        threshold = st.slider("Decision threshold (Fraud if probability ≥ threshold)",
                              min_value=0.05, max_value=0.95, value=0.50, step=0.01)

    # Construct input row
    raw = pd.DataFrame([{
        "AGE": int(AGE),
        "GENDER": GENDER.upper(),
        "DIAGNOSIS": DIAGNOSIS,
        "ENCOUNTER_DAY": ENCOUNTER_DAY,
        "LENGTH_OF_STAY": int(LENGTH_OF_STAY),
        "Amount Billed": float(AMOUNT_BILLED),
    }])

    st.caption("Input preview")
    st.dataframe(raw, use_container_width=True)

    if st.button("Predict"):
        try:
            X = preprocess_df(raw, preprocessor=preprocessor, scaler=scaler, model=model)
            proba = float(predict_proba(model, X)[0])
            label = "Fraud" if proba >= threshold else "Non-Fraud"

            if label == "Fraud":
                st.error(f"Prediction: **{label}** • Probability: **{proba:.2%}**")
            else:
                st.success(f"Prediction: **{label}** • Probability (Fraud): **{proba:.2%}**")

            with st.expander("Debug / Features fed to model"):
                if isinstance(X, pd.DataFrame):
                    st.write("Columns:", list(X.columns))
                    st.write("Row values:", X.iloc[0].to_dict())
                else:
                    st.write("Feature vector shape:", np.array(X).shape)

            with st.expander("Explain prediction (SHAP)", expanded=False):
                show_shap_explanation(model, X if isinstance(X, pd.DataFrame) else pd.DataFrame(X))
        except Exception as e:
            st.exception(e)
            st.warning("Check that your preprocessor/scaler/model were trained with the same feature schema.")

    st.markdown("---")
    st.markdown("### Batch Scoring (CSV)")
    st.caption("Upload a CSV with the same columns used in training (e.g., AGE, GENDER, DIAGNOSIS, ENCOUNTER_DAY, LENGTH_OF_STAY, Amount Billed).")
    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is not None:
        try:
            df_in = pd.read_csv(up)
            Xb = preprocess_df(df_in, preprocessor=preprocessor, scaler=scaler, model=model)
            probs = predict_proba(model, Xb)
            labels = (probs >= threshold).astype(int)

            out = df_in.copy()
            out["fraud_proba"] = probs
            out["label"] = np.where(labels == 1, "Fraud", "Non-Fraud")

            st.success(f"Scored {len(out):,} rows.")
            st.dataframe(out.head(20), use_container_width=True)
            download_button_from_df(out, filename="fraud_scored.csv")
        except Exception as e:
            st.exception(e)
            st.warning("Make sure your CSV columns match the training schema (names + types).")

    st.markdown(
        f"""
        <div style="margin-top:18px; text-align:center; opacity:.65;">
            Built with Streamlit • Themed in <b style="color:{PRIMARY}">Daewoong Orange</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
