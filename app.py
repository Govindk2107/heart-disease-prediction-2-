import os

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Dataset me jis class ko "heart disease" maana jaye (usually 1).
# Agar tumhare heart.csv me result ulta aaye to isse 0 kar dena.
POSITIVE_CLASS = 1
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart.csv")


@st.cache_data
def load_data(path_or_file):
    return pd.read_csv(path_or_file)


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


st.title("❤️ Heart Disease Prediction")
st.caption("Neeche saari details bharo aur 'Predict' dabao. Ye sirf ek ML demo hai, doctor ki salah ka vikalp nahi.")

# ---------- Data load ----------
if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
else:
    uploaded = st.file_uploader("heart.csv upload karo", type="csv")
    if uploaded is None:
        st.info("app.py ke same folder me heart.csv rakho ya yahan upload karo.")
        st.stop()
    df = load_data(uploaded)

model, accuracy = train_model(df)
st.sidebar.metric("Model accuracy (test data)", f"{accuracy * 100:.1f}%")
st.sidebar.write(f"Dataset rows: {len(df)}")

# ---------- Input form ----------
st.subheader("Patient ki information")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (umar, saal me) — age", 1, 120, 50)

        sex = st.selectbox(
            "Gender (ling) — sex",
            options=[1, 0],
            format_func=lambda x: "Male (1)" if x == 1 else "Female (0)",
        )

        cp = st.selectbox(
            "Chest Pain Type (seene ka dard) — cp",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 - Typical Angina",
                1: "1 - Atypical Angina",
                2: "2 - Non-anginal Pain",
                3: "3 - Asymptomatic (dard nahi)",
            }[x],
        )

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg) — trestbps", 50, 250, 120
        )

        chol = st.number_input("Cholesterol (mg/dl) — chol", 100, 600, 240)

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl — fbs",
            options=[0, 1],
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        )

        restecg = st.selectbox(
            "Resting ECG Result — restecg",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 - Normal",
                1: "1 - ST-T wave abnormality",
                2: "2 - Left ventricular hypertrophy",
            }[x],
        )

    with col2:
        thalach = st.number_input(
            "Max Heart Rate Achieved (heart rate, bpm) — thalach", 60, 220, 150
        )

        exang = st.selectbox(
            "Exercise Induced Angina (exercise me seene ka dard) — exang",
            options=[0, 1],
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
        )

        oldpeak = st.number_input(
            "ST Depression (exercise ke baad) — oldpeak",
            0.0, 10.0, 1.0, step=0.1,
        )

        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment — slope",
            options=[0, 1, 2],
            format_func=lambda x: {0: "0 - Upsloping", 1: "1 - Flat", 2: "2 - Downsloping"}[x],
        )

        ca = st.selectbox(
            "Major Vessels coloured by Fluoroscopy (0-4) — ca",
            options=[0, 1, 2, 3, 4],
        )

        thal = st.selectbox(
            "Thalassemia — thal",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 - Null / Unknown",
                1: "1 - Fixed Defect",
                2: "2 - Normal",
                3: "3 - Reversible Defect",
            }[x],
        )

    submitted = st.form_submit_button("🔍 Predict")

# ---------- Prediction ----------
if submitted:
    input_df = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal]],
        columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    )
    # Columns ka order dataset jaisa hi rakho
    input_df = input_df[[c for c in df.columns if c != "target"]]

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    risk = proba[list(model.classes_).index(POSITIVE_CLASS)]

    st.subheader("Result")
    if pred == POSITIVE_CLASS:
        st.error(f"⚠️ Heart disease ka risk ZYADA hai ({risk * 100:.1f}%)")
    else:
        st.success(f"✅ Heart disease ka risk KAM hai ({risk * 100:.1f}%)")
    st.progress(float(risk))
    st.caption("Ye prediction medical diagnosis nahi hai. Kisi cardiologist se zaroor milo.")