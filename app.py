# app.py — Streamlit Heart Disease Risk App (full)
import os
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load

# Components
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from streamlit_echarts import st_echarts
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="Heart Risk", page_icon="❤️", layout="wide")

# -----------------------------
# Simple authentication via secrets (optional)
# -----------------------------
creds = st.secrets.get("credentials", {})

def login():
    if not creds:
        return True, {"name": "Guest"}  # No secrets configured → skip login
    if "auth" not in st.session_state:
        st.session_state.auth = {"ok": False}
    with st.sidebar:
        st.subheader("Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Sign in"):
            usernames = creds.get("usernames", [])
            passwords = creds.get("passwords", [])
            names = creds.get("names", [])
            if u in usernames:
                i = usernames.index(u)
                if i < len(passwords) and p == passwords[i]:
                    st.session_state.auth = {"ok": True, "name": names[i] if i < len(names) else u}
                else:
                    st.error("Invalid password")
            else:
                st.error("Unknown user")
    return st.session_state.auth.get("ok", False), {"name": st.session_state.auth.get("name", "Guest")}

ok, user = login()
if not ok:
    st.stop()

# -----------------------------
# Load model safely (never st.stop)
# -----------------------------
@st.cache_resource
def load_model():
    here = os.path.dirname(__file__)
    model_path = os.path.join(here, "model.pkl")
    if not os.path.exists(model_path):
        return None, {"feature_order": []}
    obj = load(model_path)
    return obj["model"], obj["meta"]

model, meta = load_model()
FEATURES = meta.get("feature_order", [])

# -----------------------------
# Sidebar navigation (component #2: option_menu)
# -----------------------------
with st.sidebar:
    choice = option_menu(
        menu_title="Heart Risk App",
        options=["Single Prediction", "Batch Predict", "Model Card", "About"],
        icons=["activity", "table", "bar-chart", "info-circle"],
        menu_icon="heart",
        default_index=0,
    )

# Header
st.title("❤️ Heart Disease Risk — ML Demo")
st.caption(f"Welcome, {user['name']}. Enter data or upload a CSV to get predictions. (Educational demo only)")

# Tiny lottie placeholder (component #5)
def lottie_heart():
    return {"v":"5.5.7","fr":30,"ip":0,"op":60,"w":200,"h":200,"layers":[]}

# -----------------------------
# Single Prediction
# -----------------------------
if choice == "Single Prediction":
    st.subheader("Enter Patient Features")
    # Fallback feature list if model missing
    feats = FEATURES if FEATURES else ["age", "sex", "cp", "trestbps", "chol", "thalach", "oldpeak"]
    with st.form("single_form"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(feats):
            with cols[i % 3]:
                values[feat] = st.number_input(feat, value=0.0)
        submitted = st.form_submit_button("Predict Risk")
    if submitted:
        if model is None or not FEATURES:
            st.warning("No trained model loaded (model.pkl). Train and save first, then retry.")
        else:
            X = pd.DataFrame([values])[FEATURES]
            proba = float(model.predict_proba(X)[:, 1][0])
            pred = int(proba >= 0.5)
            st.metric("Risk Probability", f"{proba:.2%}")
            st.success("High risk" if pred == 1 else "Low risk")
            st_lottie(lottie_heart(), height=120)

# -----------------------------
# Batch Predict (component #3: AG Grid)
# -----------------------------
elif choice == "Batch Predict":
    st.subheader("Upload CSV or Edit Table")
    if not FEATURES:
        st.info("Model features unknown. Train the model first (model.pkl) to enable batch prediction.")
    st.write("Expected columns:", ", ".join(FEATURES) if FEATURES else "(unknown)")
    # Start with template 1-row dataframe
    df = pd.DataFrame([{f: 0.0 for f in (FEATURES if FEATURES else ['age','sex','cp'])}])

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            if FEATURES:
                df = df[FEATURES]  # enforce order and subset
        except Exception as e:
            st.error(f"Invalid CSV: {e}")
            df = pd.DataFrame([{f: 0.0 for f in (FEATURES if FEATURES else ['age','sex','cp'])}])

    gob = GridOptionsBuilder.from_dataframe(df)
    gob.configure_default_column(editable=True, resizable=True)
    grid = AgGrid(df, gridOptions=gob.build(), height=280)
    edited = grid["data"] if isinstance(grid, dict) else df

    if st.button("Run Batch Prediction"):
        if model is None or not FEATURES:
            st.warning("No trained model loaded (model.pkl). Train and save first, then retry.")
        else:
            X = pd.DataFrame(edited)[FEATURES]
            prob = model.predict_proba(X)[:, 1]
            yhat = (prob >= 0.5).astype(int)
            out = X.copy()
            out["risk_proba"] = prob
            out["prediction"] = yhat
            st.dataframe(out)
            st.download_button(
                "Download predictions.csv",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
            )

# -----------------------------
# Model Card (component #4: ECharts bar)
# -----------------------------
elif choice == "Model Card":
    st.subheader("Model Card")
    st.markdown("""
    **Model:** Logistic Regression (scikit-learn)  
    **Target:** Probability of heart disease (binary)  
    **Data:** Local `heartdataset.csv`  
    **Preprocessing:** Standardize numeric features; median-impute NAs  
    **Threshold:** 0.5 for classification  
    **Use:** Educational demo — not medical advice
    """)
    if model is None or not FEATURES:
        st.info("Train and save `model.pkl` to view feature importance.")
    else:
        try:
            clf = model.named_steps["clf"]
            coefs = np.abs(clf.coef_[0])
            data = sorted(
                [{"value": float(v), "name": FEATURES[i]} for i, v in enumerate(coefs)],
                key=lambda x: x["value"], reverse=True
            )[:10]
            option = {
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "value"},
                "yAxis": {"type": "category", "data": [d["name"] for d in data]},
                "series": [{"type": "bar", "data": [d["value"] for d in data]}],
            }
            st_echarts(options=option, height="420px")
        except Exception as e:
            st.warning(f"Could not display feature chart: {e}")

# -----------------------------
# About
# -----------------------------
elif choice == "About":
    st.markdown("""
    ### About This App
    - Built with **Streamlit** + **scikit-learn**
    - Includes 5 community components:
      - `streamlit-authenticator` (auth via secrets)
      - `streamlit-option-menu` (sidebar nav)
      - `st-aggrid` (editable batch table)
      - `streamlit-echarts` (feature bar chart)
      - `streamlit-lottie` (animation)
    - Created for HBAI program deployment assignment
    """)


