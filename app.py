from fastapi import FastAPI
from pydantic import BaseModel  
import joblib
import numpy as np

#Load model 

model = joblib.load("models/heart_disease_model.pkl")
feature_names =joblib.load("models/heart_disease_features.pkl")

app = FastAPI(title = "Heart Disease Prediction API")

#Input schema 

class PatientData(BaseModel):
    age: float 
    sex: float 
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float  
    exang: float
    oldpeak: float
    slope: float  
    ca: float
    thal: float

    #Health check endpoint

@app.get("/")
def root ():
    return {"status": "Heart Disease Prediction API is running"}



    # Prediction endpoint

@app.post("/predict")
def predict(data: PatientData):
        # Convert input data to numpy array 

  input_array= np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]])

        # Make prediction using the loaded model

  prediction = model.predict(input_array)[0]
  probability = model.predict_proba(input_array)[0][1]

  return {"prediction": int(prediction), 
          "result": "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected",
          
          "confidence": round(float(probability)*100, 2)
          }