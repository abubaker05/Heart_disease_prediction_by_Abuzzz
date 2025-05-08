import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('model/heart_disease_model.pkl')

# App configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.write("Fill in the following information to predict your heart disease risk.")

# Input form
def user_input_features():
    age = st.slider('Age', 20, 100, 50)

    sex = st.radio('Gender', ['Male', 'Female'])
    sex = 1 if sex == 'Male' else 0

    cp = st.selectbox('Chest Pain Type', [
        'Typical Angina',
        'Atypical Angina',
        'Non-anginal Pain',
        'Asymptomatic'
    ])
    cp_dict = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    cp = cp_dict[cp]

    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)

    fbs = st.radio('Fasting Blood Sugar > 120 mg/dl?', ['Yes', 'No'])
    fbs = 1 if fbs == 'Yes' else 0

    restecg = st.selectbox('Resting Electrocardiographic Results', [
        'Normal', 'Having ST-T wave abnormality', 'Left ventricular hypertrophy'
    ])
    restecg_dict = {'Normal': 0, 'Having ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    restecg = restecg_dict[restecg]

    thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)

    exang = st.radio('Exercise Induced Angina?', ['Yes', 'No'])
    exang = 1 if exang == 'Yes' else 0

    oldpeak = st.slider('ST depression induced by exercise', 0.0, 6.0, 1.0)

    slope = st.selectbox('Slope of peak exercise ST segment', [
        'Upsloping', 'Flat', 'Downsloping'
    ])
    slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope = slope_dict[slope]

    ca = st.selectbox('Number of major vessels colored by fluoroscopy', [0, 1, 2, 3])

    thal = st.selectbox('Thalassemia', [
        'Normal', 'Fixed Defect', 'Reversible Defect'
    ])
    thal_dict = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    thal = thal_dict[thal]

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Capture user input
input_df = user_input_features()

# Predict
if st.button('Predict Heart Disease Risk'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease. [Risk Probability: {prediction_proba[0][1]*100:.2f}%]")
        st.info("👉 Recommendation: Please consult a cardiologist immediately. Adopt a healthy diet, regular exercise, and monitor cholesterol levels.")
    else:
        st.success(f"✅ Low Risk of Heart Disease. [Confidence: {prediction_proba[0][0]*100:.2f}%]")
        st.info("👍 Recommendation: Maintain a healthy lifestyle with regular health checkups.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>"
    "Developed by <b>Mohammed Abu Baker</b> | © 2025"
    "</div>",
    unsafe_allow_html=True
)