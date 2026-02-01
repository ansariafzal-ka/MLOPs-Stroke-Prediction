# app.py (Streamlit UI)
import streamlit as st
import requests
import json

st.title("🩺 Stroke Risk Prediction")
st.write("Enter patient details to assess stroke risk")

# Input form
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        
    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Other"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    
    submitted = st.form_submit_button("Predict Stroke Risk")

# When form is submitted
if submitted:
    # Prepare data
    patient_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "smoking_status": smoking
    }
    
    # Call FastAPI endpoint
    try:
        response = requests.post("http://localhost:8000/predict", json=patient_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            if result["prediction"] == 1:
                st.error(f"🚨 HIGH STROKE RISK Detected!")
            else:
                st.success(f"✅ LOW Stroke Risk")
            
            st.metric("Stroke Probability", f"{result['probability']*100:.1f}%")
            
            # Risk indicator
            st.progress(result["probability"])
            
            # Details
            with st.expander("See details"):
                st.json(result)
        else:
            st.error(f"API Error: {response.text}")
            
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.info("Make sure FastAPI server is running: `uvicorn app:app --reload`")