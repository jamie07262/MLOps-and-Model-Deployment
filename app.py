from contextlib import asynccontextmanager
import os
from typing import Any, Dict, List
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator


MODEL_PATH = os.getenv("MODEL_PATH", "models/linear_regression_artifact.pkl")

model = None
model_name = "taxi-tip-regressor"
model_version = "1.0.0"
training_metrics: Dict[str, Any] = {}

FEATURES = [
    "passenger_count", "payment_type", "fare_amount", "extra", "mta_tax", "tolls_amount",
    "improvement_surcharge", "congestion_surcharge", "Airport_fee", "trip_duration_minutes",
    "trip_speed_mph", "log_trip_distance", "fare_per_mile", "fare_per_minute",
    "pickup_borough_Bronx", "pickup_borough_Brooklyn", "pickup_borough_EWR",
    "pickup_borough_Manhattan", "pickup_borough_N/A", "pickup_borough_Queens",
    "pickup_borough_Staten Island", "pickup_borough_Unknown", "dropoff_borough_Bronx",
    "dropoff_borough_Brooklyn", "dropoff_borough_EWR", "dropoff_borough_Manhattan",
    "dropoff_borough_N/A", "dropoff_borough_Queens", "dropoff_borough_Staten Island",
    "dropoff_borough_Unknown"
]


class PredictionInput(BaseModel):
    passenger_count: int = Field(..., ge=1, le=6)
    payment_type: int = Field(..., ge=0, le=6)
    fare_amount: float = Field(..., ge=0)
    extra: float = Field(..., ge=0)
    mta_tax: float = Field(..., ge=0)
    tolls_amount: float = Field(..., ge=0)
    improvement_surcharge: float = Field(..., ge=0)
    congestion_surcharge: float = Field(..., ge=0)
    Airport_fee: float = Field(..., ge=0)
    trip_duration_minutes: float = Field(..., gt=0)
    trip_speed_mph: float = Field(..., gt=0)
    log_trip_distance: float = Field(..., ge=0)
    fare_per_mile: float = Field(..., ge=0)
    fare_per_minute: float = Field(..., ge=0)

    pickup_borough_Bronx: int = Field(..., ge=0, le=1)
    pickup_borough_Brooklyn: int = Field(..., ge=0, le=1)
    pickup_borough_EWR: int = Field(..., ge=0, le=1)
    pickup_borough_Manhattan: int = Field(..., ge=0, le=1)
    pickup_borough_N_A: int = Field(..., alias="pickup_borough_N/A", ge=0, le=1)
    pickup_borough_Queens: int = Field(..., ge=0, le=1)
    pickup_borough_Staten_Island: int = Field(..., alias="pickup_borough_Staten Island", ge=0, le=1)
    pickup_borough_Unknown: int = Field(..., ge=0, le=1)

    dropoff_borough_Bronx: int = Field(..., ge=0, le=1)
    dropoff_borough_Brooklyn: int = Field(..., ge=0, le=1)
    dropoff_borough_EWR: int = Field(..., ge=0, le=1)
    dropoff_borough_Manhattan: int = Field(..., ge=0, le=1)
    dropoff_borough_N_A: int = Field(..., alias="dropoff_borough_N/A", ge=0, le=1)
    dropoff_borough_Queens: int = Field(..., ge=0, le=1)
    dropoff_borough_Staten_Island: int = Field(..., alias="dropoff_borough_Staten Island", ge=0, le=1)
    dropoff_borough_Unknown: int = Field(..., ge=0, le=1)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_tip_amount: float = Field(..., json_schema_extra={"example": 3.81})
    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    prediction_id: str = Field(..., json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"})


class BatchPredictionInput(BaseModel):
    records: List[PredictionInput] = Field(..., description="List of prediction records (max 100)")

    @field_validator("records")
    @classmethod
    def check_batch_size(cls, value: List[PredictionInput]) -> List[PredictionInput]:
        if not (1 <= len(value) <= 100):
            raise ValueError("Batch size must be between 1 and 100.")
        return value


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    api_status: str = Field(..., json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    feature_names: List[str]
    training_metrics: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_name, model_version, training_metrics

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    model_name = artifact.get("model_name", "taxi-tip-regressor")
    model_version = artifact.get("model_version", "1.0.0")
    training_metrics = artifact.get("metrics", {})

    print(f"Model loaded successfully from {MODEL_PATH}")
    yield
    print("Application shutting down.")


app = FastAPI(
    title="Taxi Tip Prediction API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", summary="Root endpoint", response_model=dict, tags=["Root"])
def root():
    """
    Root endpoint for health/status check.

    Returns:
        dict: A message indicating the API is running.
    """
    return {"message": "Taxi Tip Prediction API is running"}


@app.post("/predict", summary="Make a single tip prediction", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    try:
        x_input = pd.DataFrame([input_data.model_dump(by_alias=True)])
        x_input = x_input[FEATURES]
        prediction = model.predict(x_input)

        return PredictionResponse(
            predicted_tip_amount=round(float(prediction[0]), 2),
            model_version=model_version,
            prediction_id=str(uuid.uuid4())
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction."
        )


@app.post("/predict/batch", summary="Make batch tip predictions", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionInput):
    try:
        x_input = pd.DataFrame([record.model_dump(by_alias=True) for record in batch.records])
        x_input = x_input[FEATURES]
        predictions = model.predict(x_input)

        results = [
            PredictionResponse(
                predicted_tip_amount=round(float(pred), 2),
                model_version=model_version,
                prediction_id=str(uuid.uuid4())
            )
            for pred in predictions
        ]

        return BatchPredictionResponse(predictions=results)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during batch prediction."
        )


@app.get("/health", summary="Check API health", response_model=HealthResponse)
def health():
    return HealthResponse(
        api_status="ok",
        model_loaded=model is not None,
        model_version=model_version
    )


@app.get("/model/info", summary="Get model information", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=model_name,
        model_version=model_version,
        feature_names=FEATURES,
        training_metrics=training_metrics
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )