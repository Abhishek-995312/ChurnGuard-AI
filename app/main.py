from fastapi import FastAPI

from app.schemas import CustomerInput, PredictionResponse
from app.services.prediction import predict_churn

app = FastAPI(
    title="ChurnGuard AI",
    description="Customer churn prediction API",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ChurnGuard AI",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    return predict_churn(customer.model_dump())