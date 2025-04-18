from fastapi import FastAPI, Query
from pydantic import BaseModel 
from typing import Optional
from enum import Enum
from modelTest import svm_classification_with_plot
from modelSave import predict as model_predict 

app = FastAPI()

class MetricName(str, Enum):
    accuracy = "accuracy"
    confusion_matrix = "confusion_matrix"
    classification_report = "classification_report"
    all = "all"

class PredictRequest(BaseModel):
    experience_years: int
    technical_score: int

@app.get("/")
def home():
    return {"message": "SVM API is running"}

@app.get("/svm_report")
def get_svm_metrics(
    metric: MetricName = Query(..., description="Select metric to return")
):
    results = svm_classification_with_plot(return_metrics=True, show_plot=False)

    if metric == MetricName.all:
        return results
    elif metric in results:
        return {metric: results[metric]}
    else:
        return {"error": "Metric not found"}

@app.post("/predict")
def predict_endpoint(payload: PredictRequest):

    #POST: İşe alım tahmini yapar (0 ya da 1)
    
    prediction = model_predict(payload.experience_years, payload.technical_score)
    return {
        "experience_years": payload.experience_years,
        "technical_score": payload.technical_score,
        "prediction": int(prediction[0])
    }
