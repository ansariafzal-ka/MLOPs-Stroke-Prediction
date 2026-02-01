import pandas as pd
import numpy as np
import pickle

with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def preprocess_input(patient_dict):
    df = pd.DataFrame([patient_dict])
    
    # Your preprocessing steps
    df['bmi'] = np.log1p(df['bmi'])
    df['avg_glucose_level'] = np.log1p(df['avg_glucose_level'])
    df['over_40'] = (df['age'] > 40).astype(int)
    df['medical_risk_score'] = df['hypertension'] + df['heart_disease']
    df['gender_Male'] = (df['gender'] == 'Male').astype(int)
    df['Residence_type_Urban'] = (df['Residence_type'] == 'Urban').astype(int)
    df['ever_married'] = (df['ever_married'] == 'Yes').astype(int)
    df['work_type'] = df['work_type'].map({'Self-employed': 2, 'Private': 1, 'Govt_job': 1, 'Other': 0})
    df['smoking_status'] = df['smoking_status'].map({'formerly smoked': 2, 'smokes': 1, 'never smoked': 0, 'Unknown': 0})
    
    df = df.drop(['gender', 'Residence_type'], axis=1)
    df[['age', 'bmi', 'avg_glucose_level']] = scaler.transform(df[['age', 'bmi', 'avg_glucose_level']])
    
    return df