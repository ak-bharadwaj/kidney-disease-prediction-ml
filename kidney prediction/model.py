import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------- Load Artifacts ----------------------
model = joblib.load("ensemble_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# ---------------------- UI ----------------------
st.set_page_config(page_title="CKD Prediction", layout="centered")
st.title("🩺 Chronic Kidney Disease Risk Prediction")
st.write("Enter your medical details to calculate CKD probability. This is for awareness only, not a medical diagnosis.")

st.markdown("### Please enter your details:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 1, 100, 45)
    bp = st.number_input("Blood Pressure (mmHg)", 50, 180, 80)
    sg = st.number_input("Specific Gravity (1.005–1.025)", 1.005, 1.025, 1.015, step=0.001)
    al = st.number_input("Albumin (0–5)", 0, 5, 0)
    su = st.number_input("Sugar (0–5)", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random (mg/dL)", 50, 500, 120)
    bu = st.number_input("Blood Urea (mg/dL)", 1, 300, 40)

with col2:
    sc = st.number_input("Serum Creatinine (mg/dL)", 0.1, 20.0, 1.2, step=0.1)
    sod = st.number_input("Sodium (mEq/L)", 100, 200, 140)
    pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5, step=0.1)
    hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 17.0, 13.0, step=0.1)
    pcv = st.number_input("Packed Cell Volume (%)", 20, 60, 40)
    wc = st.number_input("White Blood Cell Count", 2000, 20000, 8000)
    rc = st.number_input("Red Blood Cell Count (millions/cmm)", 2.0, 8.0, 5.0, step=0.1)

# ---------------------- Prediction ----------------------
def predict_ckd():
    data = pd.DataFrame([{
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su, "bgr": bgr, "bu": bu,
        "sc": sc, "sod": sod, "pot": pot, "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc
    }])

    # Feature engineering
    data["kidney_function_index"] = np.log(data["sc"] * data["bu"] + 1)
    data["hydration_index"] = data["sod"] / (data["pot"] + 1e-6)
    data["anemia_risk"] = (15 - data["hemo"]) / (data["age"] + 1e-6)
    data["urine_protein_index"] = data["al"] * data["sg"]
    data["bp_age_ratio"] = data["bp"] / (data["age"] + 1e-6)

    # Select features
    data = data[selected_features]

    # Scale
    data_scaled = scaler.transform(data)

    # Predict
    pred = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0][1]

    return pred, proba, data

# ---------------------- Button ----------------------
if st.button("🔍 Predict CKD Risk"):
    pred, proba, df_input = predict_ckd()

    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk of CKD (Probability: {proba:.2%})")
    else:
        st.success(f"✅ Low Risk of CKD (Probability: {proba:.2%})")

    st.markdown("### 📝 Your Inputs")
    st.dataframe(df_input.T, use_container_width=True)

    st.markdown("### 📈 Probability Chart")
    fig, ax = plt.subplots()
    ax.bar(["No CKD", "CKD"], [1 - proba, proba], color=["green", "red"])
    ax.set_ylabel("Probability")
    st.pyplot(fig)
