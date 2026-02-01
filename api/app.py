from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import sys
import pandas as pd

from src.exception import CustomException
from src.utils.preprocess import preprocess_input

app = FastAPI()

class PatientData(BaseModel):
    gender: object
    age: float
    hypertension: int
    heart_disease: int
    ever_married: object
    work_type: object
    Residence_type: object
    avg_glucose_level: float
    bmi: float
    smoking_status: object

@app.get('/')
def root():
    return {'status': 'ok'}

@app.post('/predict')
def predict(patient: PatientData):
    try:
        with open('artifacts/model.pkl', 'rb') as f:
            model = pickle.load(f)

        processed_df = preprocess_input(patient.dict())
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0][1]

        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk': 'High' if probability > 0.5 else 'Low'
        }

    except Exception as e:
        raise CustomException(e, sys)