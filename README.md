# Heart Disease Risk – Streamlit App (Local CSV)

An interactive Streamlit app that serves a Logistic Regression model trained on your **local** heart disease dataset.

---

## 📋 Overview

This app predicts the likelihood of heart disease using a Logistic Regression model trained on your local CSV file (`heartdataset.csv`). The app provides both **single prediction** and **batch prediction** modes, visual explanations, and authentication support.

---

## 📦 Project Structure

```
Heart-Disease-Risk---Streamlit-App/
├─ app.py                     # Streamlit web app (main entry)
├─ train_model.py             # Script to train model and create model.pkl
├─ model.pkl                  # Saved trained model
├─ heartdataset.csv           # ✅ Local dataset
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## ✅ Key Streamlit Components (All from Different Categories)

1. **Authentication** → `streamlit-authenticator`
2. **Navigation Layout** → `streamlit-option-menu`
3. **Data Editing / Display** → `st-aggrid`
4. **Charts / Visualization** → `streamlit-echarts`
5. **Media / Animation** → `streamlit-lottie`

---

## 🧠 Model Summary

* **Task:** Binary classification – risk of heart disease.
* **Algorithm:** Logistic Regression (scikit-learn)
* **Dataset:** Local CSV file `heartdataset.csv`
* **Metrics (Example):** F1 ≈ 0.80±, ROC‑AUC ≈ 0.85± (depending on your data split)

---

## 🔧 Setup Instructions

### 1️⃣ Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # On macOS/Linux
# OR
.venv\Scripts\activate          # On Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model (Required Once)

Make sure your dataset exists at this path:

```
/Users/user/Documents/GitHub/Heart-Disease-Risk---Streamlit-App/heartdataset.csv
```

Then run:

```bash
python train_model.py
```

This generates `model.pkl` in the same directory.

### 4️⃣ Launch the App

```bash
streamlit run app.py
```

The app will open in your browser (usually at [http://localhost:8501](http://localhost:8501)).

---

## 🧾 requirements.txt

```txt
streamlit==1.39.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.4.2
streamlit-authenticator==0.3.2
streamlit-option-menu==0.3.12
streamlit-echarts==0.4.0
streamlit-lottie==0.0.5
streamlit-aggrid==0.3.4.post3
```

---

## 🔐 Authentication Setup

Create a `.streamlit/secrets.toml` file in your project directory (and add it to **Streamlit Cloud Secrets** if deploying):

```toml
[credentials]
usernames = ["guneet"]
passwords = ["pass1234"]
names = ["Guneet Singh"]

[cookie]
name = "heart_auth"
key = "a-very-secret-cookie-key"
expiry_days = 3
```

This allows a simple login before using the app.

---

## 💾 Data File Location

**Local CSV Path:**

```
/Users/user/Documents/GitHub/Heart-Disease-Risk---Streamlit-App/heartdataset.csv
```

You can change this in `train_model.ipynb` under:

```python
DATA_PATH = "/Users/user/Documents/GitHub/Heart-Disease-Risk---Streamlit-App/heartdataset.csv"
```

If deploying to Streamlit Cloud, commit the dataset file to your repo or set an environment variable:

```bash
export HEART_CSV_PATH="/mount/path/to/heartdataset.csv"
```

---



Built by **Guneet Singh** for Durham College’s **HBAI Program Streamlit ML Deployment Assignment (2025)**.

**Dataset:** Heart Disease UCI Dataset (adapted to local file)
**Libraries:** Streamlit, scikit-learn, pandas, numpy, joblib
**Purpose:** Educational / Demo Only — *not for medical diagnosis*.
