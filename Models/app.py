# app.py
# Streamlit demo for Healthcare Fraud Detection (NHIS Healthcare Claims & Fraud)
# Dataset source: Kaggle (bonifacechosen / NHIS Healthcare Claims & Fraud)

import os
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional/model-specific deps
try:
    import joblib
except Exception:
    joblib = None

try:
    import shap  # optional
except Exception:
    shap = None

try:
    import matplotlib.pyplot as plt  # for SHAP matplotlib backend
except Exception:
    plt = None


# =============================
# THEME (Daewoong vibes)
# =============================
PRIMARY = "#F26B21"     # Daewoong orange
PRIMARY_DARK = "#C45319"
BG = "#0F1117"
CARD = "#1B1F2A"
TEXT = "#E6E6E6"

st.set_page_config(page_title="Healthcare Fraud Detection", page_icon="🧾", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background:{BG}; color:{TEXT}; }}
    section[data-testid="stSidebar"] {{ background:#161A23; border-right:1px solid #232838; }}
    .daewoong-hero {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        color: white; padding: 28px 36px; border-radius: 18px; text-align: center;
        box-shadow: 0 10px 24px rgba(0,0,0,.45); margin-bottom: 24px; border: 1px solid rgba(255,255,255,.08);
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
# HELPERS
# =============================
BASE_DIR = Path(__file__).resolve().parent  # always read from app folder

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

def _safe_load(p: Path):
    """Try joblib first, then pickle; return None if both fail with warning."""
    obj = None
    err1 = err2 = None
    if joblib is not None:
        try:
            obj = joblib.load(p)
            return obj
        except Exception as e:
            err1 = e
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
            return obj
    except Exception as e:
        err2 = e
    st.warning(f"Failed to load {p.name}. joblib_error={err1} pickle_error={err2}")
    return None

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load models + encoders/scalers/metadata with robust fallbacks."""
    # Accept these model filenames if present
    model_files = {
        "XGBoost": "model_xgb.pkl",
        "LightGBM": "model_lgbm.pkl",
        "Random Forest": "model_rf.pkl",
        "Logistic Regression": "model_logreg.pkl",
        "CatBoost": "model_cat.pkl",
        "Model": "model.pkl",  # generic
    }
    models = {}
    for name, fn in model_files.items():
        p = BASE_DIR / fn
        if p.exists():
            obj = _safe_load(p)
            if obj is not None:
                models[name] = obj

    # Preprocessor/encoder (accept several common names; first found wins)
    preprocessor = None
    for fn in ["preprocessor.pkl", "encoder.pkl", "transformer.pkl", "target_encoder.pkl"]:
        p = BASE_DIR / fn
        if p.exists():
            preprocessor = _safe_load(p)
            if preprocessor is not None:
                break

    # Scaler: also handle names like "scaler (3).pkl"
    scaler = None
    for p in list(BASE_DIR.glob("scaler*.pkl")) + [BASE_DIR / "standard_scaler.pkl"]:
        if p.exists():
            scaler = _safe_load(p)
            if scaler is not None:
                break

    # Optional: leaderboard & metrics
    comp_df = None
    p = BASE_DIR / "model_comparison.csv"
    if p.exists():
        try:
            comp_df = pd.read_csv(p)
        except Exception as e:
            st.warning(f"Could not read model_comparison.csv: {e}")

    metrics = None
    p = BASE_DIR / "metrics.json"
    if p.exists():
        try:
            metrics = json.loads(p.read_text())
        except Exception as e:
            st.warning(f"Could not read metrics.json: {e}")

    return models, preprocessor, scaler, comp_df, metrics

def preprocess_df(df, preprocessor=None, scaler=None, model=None):
    """
    Apply your saved preprocessing; otherwise fallback to pandas.get_dummies.
    Tries to scale appropriately whether scaler was fit on full matrix or just numerics.
    """
    X = df.copy()

    # 1) Encoder / ColumnTransformer
    if preprocessor is not None:
        try:
            X = preprocessor.transform(X)
        except Exception as e:
            st.warning(f"Preprocessor.transform failed: {e}")

    # 2) Scaling (best effort)
    if scaler is not None:
        try:
            if hasattr(scaler, "n_features_in_"):
                # If counts match, assume scaler was fit on the whole matrix
                if isinstance(X, pd.DataFrame) and scaler.n_features_in_ == X.shape[1]:
                    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
                elif not isinstance(X, pd.DataFrame) and scaler.n_features_in_ == np.array(X).shape[1]:
                    X = scaler.transform(X)
                else:
                    # Try scaling numeric subset in DataFrame
                    if isinstance(X, pd.DataFrame):
                        num_cols = X.select_dtypes(include=[np.number]).columns
                        if len(num_cols) == scaler.n_features_in_:
                            X[num_cols] = scaler.transform(X[num_cols])
                        else:
                            # last resort: ignore scaler mismatch
                            pass
            else:
                # Unknown scaler type; attempt numeric-only
                if isinstance(X, pd.DataFrame):
                    num_cols = X.select_dtypes(include=[np.number]).columns
                    X[num_cols] = scaler.transform(X[num_cols])
        except Exception as e:
            st.warning(f"Scaler.transform best-effort failed: {e}")

    # 3) Align to model's expected columns if available
    if model is not None and hasattr(model, "feature_names_in_"):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X = X.reindex(columns=list(model.feature_names_in_), fill_value=0)
    else:
        # Fallback to get_dummies to avoid categorical issues if we never encoded
        if isinstance(X, pd.DataFrame):
            X = pd.get_dummies(X)

    return X

def predict_proba(model, X):
    """Return probability of positive class (Fraud)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1 / (1 + np.exp(-z))
    y = model.predict(X)
    return y.astype(float)

def download_button_from_df(df, filename="scored.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download scored CSV", data=csv, file_name=filename, mime="text/csv")

def show_shap_explanation(model, X_row):
    """Local explanation with SHAP if available; safe fallbacks."""
    if shap is None or plt is None:
        st.info("Install SHAP and matplotlib (`pip install shap matplotlib`) to see local explanations.")
        return
    try:
        name = model.__class__.__name__.lower()
        if any(k in name for k in ["xgb", "lgbm", "catboost", "randomforest", "gradientboost"]):
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_row)
            base = explainer.expected_value
        else:
            # KernelExplainer can be slow; do tiny sampling
            f = lambda X: predict_proba(model, X)
            explainer = shap.KernelExplainer(f, X_row)
            values = explainer.shap_values(X_row, nsamples=100)
            base = explainer.expected_value

        shap.initjs()
        st.subheader("Local explanation (SHAP)")
        try:
            shap.force_plot(base, values[0] if isinstance(values, list) else values, X_row, matplotlib=True, show=False)
            st.pyplot(bbox_inches="tight")
        except Exception:
            arr = values[0] if isinstance(values, list) else values
            contrib = pd.Series(np.array(arr).flatten(),
                                index=getattr(X_row, "columns", range(np.array(X_row).shape[1])))
            st.bar_chart(contrib.sort_values(ascending=False).head(12))
    except Exception as e:
        st.info(f"SHAP explanation not available: {e}")


# =============================
# SIDEBAR NAV
# =============================
st.sidebar.markdown("#### Menu")
page = st.sidebar.selectbox("", ["Home", "Machine Learning App"], index=0, label_visibility="collapsed")

# Load once
models, preprocessor, scaler, comp_df, metrics = load_artifacts()

# Optional: quick debug panel
with st.expander("Debug: files & artifacts seen by the app"):
    st.write("Working dir:", os.getcwd())
    st.write("App folder:", str(BASE_DIR))
    try:
        st.write("Files in app folder:", sorted([p.name for p in BASE_DIR.iterdir()]))
    except Exception as e:
        st.write("Could not list app folder:", e)
    st.write("Loaded models:", list(models.keys()))
    st.write("Has preprocessor:", preprocessor is not None)
    st.write("Has scaler:", scaler is not None)


# =============================
# PAGES
# =============================
if page == "Home":
    hero()

    st.header("Overview")
    st.markdown(
        """
        <div class="card">
          <p><b>Objective:</b> Binary classification to flag likely <b>Fraud</b> vs <b>Non-Fraud</b> healthcare claims so SIU teams can prioritize investigation.</p>
          <p><b>Dataset:</b> NHIS Healthcare Claims & Fraud (Kaggle).</p>
          <ul>
            <li>Pipeline: Cleaning → Encoding (Target/One-Hot) → Scaling (if used) → Model.</li>
            <li>Models supported in this app: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost (load any subset).</li>
            <li>Outputs: label + probability, threshold tuning, optional SHAP explanation, batch CSV scoring.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if comp_df is not None and not comp_df.empty:
        st.subheader("Model Leaderboard (from training)")
        st.caption("Provide `model_comparison.csv` (e.g., model, roc_auc, recall, precision, f1, fit_time).")
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Place `model_comparison.csv` to show your training leaderboard here.")

    if metrics is not None:
        st.subheader("Hold-out Metrics (static)")
        st.json(metrics)
    else:
        st.caption("Optional: add `metrics.json` with summary metrics (e.g., ROC-AUC, Recall@threshold).")

    st.markdown(
        f'<div style="margin-top:18px; text-align:center; opacity:.65;">'
        f'Built with Streamlit • Themed in <b style="color:{PRIMARY}">Daewoong Orange</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

else:
    hero("Healthcare Fraud Detection — Inference", "Real-time & batch scoring")

    if not models:
        st.error("No model file found. Add `model.pkl` or one of: model_xgb.pkl, model_lgbm.pkl, model_rf.pkl, model_logreg.pkl, model_cat.pkl.")
        st.stop()

    model_name = st.selectbox("Choose model", list(models.keys()))
    model = models[model_name]

    st.markdown("### Single-case Scoring")

    # — Inputs (align to training features) —
    c1, c2 = st.columns(2)
    with c1:
        AGE = st.number_input("AGE", min_value=0, max_value=120, value=45, step=1)
    with c2:
        GENDER = st.selectbox("GENDER", ["M", "F"])

    with c1:
        DIAGNOSIS = st.text_input("DIAGNOSIS", value="Refractive Error")
    with c2:
        ENCOUNTER_DAY = st.selectbox("ENCOUNTER_DAY",
                                     ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    with c1:
        LENGTH_OF_STAY = st.number_input("LENGTH_OF_STAY (days)", min_value=0, max_value=365, value=2, step=1)
    with c2:
        AMOUNT_BILLED = st.number_input("Amount Billed", min_value=0.0, value=1500.0, step=50.0, format="%.2f")

    thr_col, _ = st.columns([1.3, 3])
    with thr_col:
        threshold = st.slider("Decision threshold (Fraud if probability ≥ threshold)", 0.05, 0.95, 0.50, 0.01)

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

            with st.expander("Explain prediction (SHAP)"):
                X_for_shap = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
                show_shap_explanation(model, X_for_shap)
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
        f'<div style="margin-top:18px; text-align:center; opacity:.65;">'
        f'Built with Streamlit • Themed in <b style="color:{PRIMARY}">Daewoong Orange</b>'
        f'</div>',
        unsafe_allow_html=True,
    )
