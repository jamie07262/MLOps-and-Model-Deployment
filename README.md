# **NYC Yellow Taxi Tip Prediction API (MLOps & Model Deployment)**

## Overview

This project implements an end-to-end MLOps pipeline for predicting taxi tip amounts. It includes model training, experiment tracking with MLflow, model serving using FastAPI, API testing, and containerized deployment using Docker and Docker Compose.

## Features

- MLflow experiment tracking and model registry
- FastAPI REST API for predictions
- Input validation using Pydantic
- Batch and single prediction endpoints
- Automated testing with pytest
- Docker containerization
- Docker Compose orchestration (API + MLflow)

## Model Serving API

The FastAPI application loads a trained model artifact at startup and reuses it across all requests (no reloading per request).

### Endpoints

| Endpoint       | Method | Description                 |
| -------------- | ------ | --------------------------- |
| /predict       | POST   | Single prediction           |
| /predict/batch | POST   | Batch predictions (max 100) |
| /health        | GET    | API health check            |
| /model/info    | GET    | Model metadata              |
| /docs          | GET    | Swagger UI                  |

## Running Locally

1. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
2. Run FastAPI
   ```sh
   uvicorn app:app --reload --port 8000
   ```
3. Access API
   - Swagger UI: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## MLflow Tracking

Start MLflow UI:

```sh
mlflow ui --port 5000
```

Access at: http://localhost:5000

## API Testing

Automated tests are implemented using pytest and FastAPI’s TestClient.

### Run tests

```sh
pytest test_app.py
```

### Test Coverage

- Successful single prediction
- Successful batch prediction (multiple records)
- Invalid input handling:
  - Missing required fields
  - Wrong data types (e.g., string instead of int)
  - Negative/invalid values (e.g., negative passenger count)
- Health check endpoint
- Model metadata endpoint
- Edge cases:
  - Zero distance trip
  - Extremely high fare values
  - Extremely low (but valid) fare values
  - Batch prediction with invalid records
- API returns correct error codes (422 for validation, 200 for success)

## Docker Containerization

### Dockerfile Summary

- Base image: python:3.11-slim
- Installs dependencies from requirements.txt
- Copies only required files (app.py, models/)
- Runs API using Uvicorn on port 8000

### Build Image

```sh
docker build -t taxi-tip-api .
```

### Run Container

```sh
docker run -p 8000:8000 taxi-tip-api
```

## Docker Compose Deployment

### Services

- api: FastAPI prediction service
- mlflow: MLflow tracking server

### Start Services

```sh
docker compose up --build
```

### Stop Services

```sh
docker compose down
```

### Access

- API: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

## Making Predictions (Example)

### PowerShell

```powershell
$body = @{
	 passenger_count = 2
	 payment_type = 1
	 fare_amount = 20.0
	 extra = 1.0
	 mta_tax = 0.5
	 tolls_amount = 0.0
	 improvement_surcharge = 1.0
	 congestion_surcharge = 2.5
	 Airport_fee = 0.0
	 trip_duration_minutes = 15.0
	 trip_speed_mph = 12.0
	 log_trip_distance = 1.2
	 fare_per_mile = 6.5
	 fare_per_minute = 1.33
	 "pickup_borough_Manhattan" = 1
	 "dropoff_borough_Manhattan" = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

## Container Details

- Image size: ~499 MB
- Port: 8000
- Model path: /app/models/linear_regression_artifact.pkl
- Environment variable: MODEL_PATH

## Environment Variables Example

You may need to set the following environment variables for local development or deployment:

```sh
MLFLOW_TRACKING_URI="http://localhost:5000"
MODEL_PATH="models/linear_regression_artifact.pkl"
```

## Requirements

All dependencies are listed in requirements.txt.

## Conclusion

This project demonstrates a complete ML deployment workflow including model tracking, API development, testing, and containerized deployment using modern MLOps practices.
