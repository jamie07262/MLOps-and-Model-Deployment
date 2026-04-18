import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200

def test_single_prediction_success(client):
    payload = {
        "passenger_count": 2,
        "payment_type": 1,
        "fare_amount": 20.0,
        "extra": 1.0,
        "mta_tax": 0.5,
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
        "trip_duration_minutes": 15.0,
        "trip_speed_mph": 12.0,
        "log_trip_distance": 1.2,
        "fare_per_mile": 6.5,
        "fare_per_minute": 1.33,
        "pickup_borough_Bronx": 0,
        "pickup_borough_Brooklyn": 0,
        "pickup_borough_EWR": 0,
        "pickup_borough_Manhattan": 1,
        "pickup_borough_N/A": 0,
        "pickup_borough_Queens": 0,
        "pickup_borough_Staten Island": 0,
        "pickup_borough_Unknown": 0,
        "dropoff_borough_Bronx": 0,
        "dropoff_borough_Brooklyn": 0,
        "dropoff_borough_EWR": 0,
        "dropoff_borough_Manhattan": 1,
        "dropoff_borough_N/A": 0,
        "dropoff_borough_Queens": 0,
        "dropoff_borough_Staten Island": 0,
        "dropoff_borough_Unknown": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_tip_amount" in data
    assert "model_version" in data
    assert "prediction_id" in data
    assert isinstance(data["predicted_tip_amount"], float)
    assert isinstance(data["model_version"], str)
    assert isinstance(data["prediction_id"], str)

def test_successful_batch_prediction(client):
    batch_payload = {
        "records": [
            {
                "passenger_count": 2,
                "payment_type": 1,
                "fare_amount": 20.0,
                "extra": 1.0,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 1.0,
                "congestion_surcharge": 2.5,
                "Airport_fee": 0.0,
                "trip_duration_minutes": 15.0,
                "trip_speed_mph": 12.0,
                "log_trip_distance": 1.2,
                "fare_per_mile": 6.5,
                "fare_per_minute": 1.33,
                "pickup_borough_Bronx": 0,
                "pickup_borough_Brooklyn": 0,
                "pickup_borough_EWR": 0,
                "pickup_borough_Manhattan": 1,
                "pickup_borough_N/A": 0,
                "pickup_borough_Queens": 0,
                "pickup_borough_Staten Island": 0,
                "pickup_borough_Unknown": 0,
                "dropoff_borough_Bronx": 0,
                "dropoff_borough_Brooklyn": 0,
                "dropoff_borough_EWR": 0,
                "dropoff_borough_Manhattan": 1,
                "dropoff_borough_N/A": 0,
                "dropoff_borough_Queens": 0,
                "dropoff_borough_Staten Island": 0,
                "dropoff_borough_Unknown": 0
            },
            {
                "passenger_count": 1,
                "payment_type": 1,
                "fare_amount": 10.0,
                "extra": 0.5,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 1.0,
                "congestion_surcharge": 2.5,
                "Airport_fee": 0.0,
                "trip_duration_minutes": 8.0,
                "trip_speed_mph": 10.0,
                "log_trip_distance": 0.8,
                "fare_per_mile": 5.0,
                "fare_per_minute": 1.25,
                "pickup_borough_Bronx": 0,
                "pickup_borough_Brooklyn": 1,
                "pickup_borough_EWR": 0,
                "pickup_borough_Manhattan": 0,
                "pickup_borough_N/A": 0,
                "pickup_borough_Queens": 0,
                "pickup_borough_Staten Island": 0,
                "pickup_borough_Unknown": 0,
                "dropoff_borough_Bronx": 0,
                "dropoff_borough_Brooklyn": 1,
                "dropoff_borough_EWR": 0,
                "dropoff_borough_Manhattan": 0,
                "dropoff_borough_N/A": 0,
                "dropoff_borough_Queens": 0,
                "dropoff_borough_Staten Island": 0,
                "dropoff_borough_Unknown": 0
            }
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert "predicted_tip_amount" in pred
        assert "model_version" in pred
        assert "prediction_id" in pred
        assert isinstance(pred["predicted_tip_amount"], float)
        assert isinstance(pred["model_version"], str)
        assert isinstance(pred["prediction_id"], str)


def test_invalid_input_for_single_prediction(client):
    invalid_payload = {
        "passenger_count": -2, # Invalid negative passenger count
        "payment_type": 1,
        "fare_amount": 20.0,
        "extra": 1.0,
        "mta_tax": 0.5,
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
        "trip_duration_minutes": 15.0,
        "trip_speed_mph": 12.0,
        "log_trip_distance": 1.2,
        "fare_per_mile": 6.5,
        "fare_per_minute": 1.33,
        "pickup_borough_Bronx": 0,
        "pickup_borough_Brooklyn": 0,
        "pickup_borough_EWR": 0,
        "pickup_borough_Manhattan": 1,
        "pickup_borough_N/A": 0,
        "pickup_borough_Queens": 0,
        "pickup_borough_Staten Island": 0,
        "pickup_borough_Unknown": 0,
        "dropoff_borough_Bronx": 0,
        "dropoff_borough_Brooklyn": 0,
        "dropoff_borough_EWR": 0,
        "dropoff_borough_Manhattan": 1,
        "dropoff_borough_N/A": 0,
        "dropoff_borough_Queens": 0,
        "dropoff_borough_Staten Island": 0,
        "dropoff_borough_Unknown": 0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

    missing_field_playload = invalid_payload.copy()
    missing_field_playload["passenger_count"] = 2
    del missing_field_playload["fare_amount"]

    response = client.post("/predict", json=missing_field_playload)
    assert response.status_code == 422

    wrong_data_payload = invalid_payload.copy()
    wrong_data_payload["passenger_count"] = "two"

    response = client.post("/predict", json=wrong_data_payload)
    assert response.status_code == 422

def test_invalid_input_for_batch_prediction(client):
    invalid_batch_payload = {
        "records": [
            {
                "passenger_count": -2,
                "payment_type": 1,
                "fare_amount": 20.0,
                "extra": 1.0,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 1.0,
                "congestion_surcharge": 2.5,
                "Airport_fee": 0.0,
                "trip_duration_minutes": 15.0,
                "trip_speed_mph": 12.0,
                "log_trip_distance": 1.2,
                "fare_per_mile": 6.5,
                "fare_per_minute": 1.33,
                "pickup_borough_Bronx": 0,
                "pickup_borough_Brooklyn": 0,
                "pickup_borough_EWR": 0,
                "pickup_borough_Manhattan": 1,
                "pickup_borough_N/A": 0,
                "pickup_borough_Queens": 0,
                "pickup_borough_Staten Island": 0,
                "pickup_borough_Unknown": 0,
                "dropoff_borough_Bronx": 0,
                "dropoff_borough_Brooklyn": 0,
                "dropoff_borough_EWR": 0,
                "dropoff_borough_Manhattan": 1,
                "dropoff_borough_N/A": 0,
                "dropoff_borough_Queens": 0,
                "dropoff_borough_Staten Island": 0,
                "dropoff_borough_Unknown": 0
            },
            {
                "passenger_count": -1,
                "payment_type": 1,
                "fare_amount": 10.0,
                "extra": 0.5,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 1.0,
                "congestion_surcharge": 2.5,
                "Airport_fee": 0.0,
                "trip_duration_minutes": 8.0,
                "trip_speed_mph": 10.0,
                "log_trip_distance": 0.8,
                "fare_per_mile": 5.0,
                "fare_per_minute": 1.25,
                "pickup_borough_Bronx": 0,
                "pickup_borough_Brooklyn": 1,
                "pickup_borough_EWR": 0,
                "pickup_borough_Manhattan": 0,
                "pickup_borough_N/A": 0,
                "pickup_borough_Queens": 0,
                "pickup_borough_Staten Island": 0,
                "pickup_borough_Unknown": 0,
                "dropoff_borough_Bronx": 0,
                "dropoff_borough_Brooklyn": 1,
                "dropoff_borough_EWR": 0,
                "dropoff_borough_Manhattan": 0,
                "dropoff_borough_N/A": 0,
                "dropoff_borough_Queens": 0,
                "dropoff_borough_Staten Island": 0,
                "dropoff_borough_Unknown": 0
            }
        ]
    }

    response = client.post("/predict/batch", json=invalid_batch_payload)
    assert response.status_code == 422

    # missing field case
    missing_field_playload = invalid_batch_payload.copy()
    missing_field_playload["records"][0]["passenger_count"] = 2
    missing_field_playload["records"][1]["passenger_count"] = 1

    del missing_field_playload["records"][0]["fare_amount"]
    del missing_field_playload["records"][1]["fare_amount"]

    response = client.post("/predict/batch", json=missing_field_playload)
    assert response.status_code == 422

    # wrong data type case
    wrong_data_payload = invalid_batch_payload.copy()
    wrong_data_payload["records"][0]["passenger_count"] = "two"
    wrong_data_payload["records"][1]["passenger_count"] = "two"
    response = client.post("/predict/batch", json=wrong_data_payload)
    assert response.status_code == 422

def test_health_check_correct_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "api_status" in data
    assert data["api_status"] == "ok"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)
    assert "model_version" in data
    assert isinstance(data["model_version"], str)

# Edge case: zero distance trip
def test_zero_distance_trip(client):
    payload = {
        "passenger_count": 1,
        "payment_type": 1,
        "fare_amount": 5.0,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
        "trip_duration_minutes": 5.0,
        "trip_speed_mph": 0.1,
        "log_trip_distance": 0.0,
        "fare_per_mile": 0.0,
        "fare_per_minute": 1.0,
        "pickup_borough_Bronx": 0,
        "pickup_borough_Brooklyn": 0,
        "pickup_borough_EWR": 0,
        "pickup_borough_Manhattan": 1,
        "pickup_borough_N/A": 0,
        "pickup_borough_Queens": 0,
        "pickup_borough_Staten Island": 0,
        "pickup_borough_Unknown": 0,
        "dropoff_borough_Bronx": 0,
        "dropoff_borough_Brooklyn": 0,
        "dropoff_borough_EWR": 0,
        "dropoff_borough_Manhattan": 1,
        "dropoff_borough_N/A": 0,
        "dropoff_borough_Queens": 0,
        "dropoff_borough_Staten Island": 0,
        "dropoff_borough_Unknown": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_tip_amount" in data
        assert "model_version" in data
        assert "prediction_id" in data

# Edge case: extreme fare values
def test_extreme_fare_values(client):
    # Extremely high fare
    high_fare_payload = {
        "passenger_count": 2,
        "payment_type": 1,
        "fare_amount": 100000.0,
        "extra": 100.0,
        "mta_tax": 50.0,
        "tolls_amount": 500.0,
        "improvement_surcharge": 100.0,
        "congestion_surcharge": 250.0,
        "Airport_fee": 50.0,
        "trip_duration_minutes": 120.0,
        "trip_speed_mph": 60.0,
        "log_trip_distance": 5.0,
        "fare_per_mile": 1000.0,
        "fare_per_minute": 100.0,
        "pickup_borough_Bronx": 0,
        "pickup_borough_Brooklyn": 0,
        "pickup_borough_EWR": 0,
        "pickup_borough_Manhattan": 1,
        "pickup_borough_N/A": 0,
        "pickup_borough_Queens": 0,
        "pickup_borough_Staten Island": 0,
        "pickup_borough_Unknown": 0,
        "dropoff_borough_Bronx": 0,
        "dropoff_borough_Brooklyn": 0,
        "dropoff_borough_EWR": 0,
        "dropoff_borough_Manhattan": 1,
        "dropoff_borough_N/A": 0,
        "dropoff_borough_Queens": 0,
        "dropoff_borough_Staten Island": 0,
        "dropoff_borough_Unknown": 0
    }
    response = client.post("/predict", json=high_fare_payload)
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_tip_amount" in data
        assert "model_version" in data
        assert "prediction_id" in data

    # Extremely low (but valid) fare
    low_fare_payload = high_fare_payload.copy()
    low_fare_payload["fare_amount"] = 1.0
    response = client.post("/predict", json=low_fare_payload)
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_tip_amount" in data
        assert "model_version" in data
        assert "prediction_id" in data







# #missing fields
# def test_missing_fields_for_single_prediction(client):
#     incomplete_payload = {
#         "payment_type": 1,
#         "fare_amount": 20.0,
#         "extra": 1.0,
#         "mta_tax": 0.5,
#         "tolls_amount": 0.0,
#         "improvement_surcharge": 1.0,
#         "congestion_surcharge": 2.5,
#         "Airport_fee": 0.0,
#         "trip_duration_minutes": 15.0,
#         "trip_speed_mph": 12.0,
#         "log_trip_distance": 1.2,
#         "fare_per_mile": 6.5,
#         "fare_per_minute": 1.33,
#         "pickup_borough_Bronx": 0,
#         "pickup_borough_Brooklyn": 0,
#         "pickup_borough_EWR": 0,
#         "pickup_borough_Manhattan": 1,
#         "pickup_borough_N/A": 0,
#         "pickup_borough_Queens": 0,
#         "pickup_borough_Staten Island": 0,
#         "pickup_borough_Unknown": 0,
#         "dropoff_borough_Bronx": 0,
#         "dropoff_borough_Brooklyn": 0,
#         "dropoff_borough_EWR": 0,
#         "dropoff_borough_Manhattan": 1,
#         "dropoff_borough_N/A": 0,
#         "dropoff_borough_Queens": 0,
#         "dropoff_borough_Staten Island": 0,
#         "dropoff_borough_Unknown": 0
#     }
#     response = client.post("/predict", json=incomplete_payload)
#     assert response.status_code == 422

# def test_missing_fields_for_batch_prediction(client):
#     incomplete_batch_payload = {
#         "records": [
#             {
#                 "payment_type": 1,
#                 "fare_amount": 20.0,
#                 "extra": 1.0,
#                 "mta_tax": 0.5,
#                 "tolls_amount": 0.0,
#                 "improvement_surcharge": 1.0,
#                 "congestion_surcharge": 2.5,
#                 "Airport_fee": 0.0,
#                 "trip_duration_minutes": 15.0,
#                 "trip_speed_mph": 12.0,
#                 "log_trip_distance": 1.2,
#                 "fare_per_mile": 6.5,
#                 "fare_per_minute": 1.33,
#                 "pickup_borough_Bronx": 0,
#                 "pickup_borough_Brooklyn": 0,
#                 "pickup_borough_EWR": 0,
#                 "pickup_borough_Manhattan": 1,
#                 "pickup_borough_N/A": 0,
#                 "pickup_borough_Queens": 0,
#                 "pickup_borough_Staten Island": 0,
#                 "pickup_borough_Unknown": 0,
#                 "dropoff_borough_Bronx": 0,
#                 "dropoff_borough_Brooklyn": 0,
#                 "dropoff_borough_EWR": 0,
#                 "dropoff_borough_Manhattan": 1,
#                 "dropoff_borough_N/A": 0,
#                 "dropoff_borough_Queens": 0,
#                 "dropoff_borough_Staten Island": 0,
#                 "dropoff_borough_Unknown": 0
#             },
#             {
#                 "payment_type": 1,
#                 "fare_amount": 10.0,
#                 "extra": 0.5,
#                 "mta_tax": 0.5,
#                 "tolls_amount": 0.0,
#                 "improvement_surcharge": 1.0,
#                 "congestion_surcharge": 2.5,
#                 "Airport_fee": 0.0,
#                 "trip_duration_minutes": 8.0,
#                 "trip_speed_mph": 10.0,
#                 "log_trip_distance": 0.8,
#                 "fare_per_mile": 5.0,
#                 "fare_per_minute": 1.25,
#                 "pickup_borough_Bronx": 0,
#                 "pickup_borough_Brooklyn": 1,
#                 "pickup_borough_EWR": 0,
#                 "pickup_borough_Manhattan": 0,
#                 "pickup_borough_N/A": 0,
#                 "pickup_borough_Queens": 0,
#                 "pickup_borough_Staten Island": 0,
#                 "pickup_borough_Unknown": 0,
#                 "dropoff_borough_Bronx": 0,
#                 "dropoff_borough_Brooklyn": 1,
#                 "dropoff_borough_EWR": 0,
#                 "dropoff_borough_Manhattan": 0,
#                 "dropoff_borough_N/A": 0,
#                 "dropoff_borough_Queens": 0,
#                 "dropoff_borough_Staten Island": 0,
#                 "dropoff_borough_Unknown": 0
#             }
#         ]
#     }

#     response = client.post("/predict/batch", json=incomplete_batch_payload)
#     assert response.status_code == 422

# #wrong data types
# def test_wrong_data_types_for_single_prediction(client):
#     wrong_type_payload = {
#         "passenger_count": "two", # Should be an integer
#         "payment_type": 1,
#         "fare_amount": 20.0,
#         "extra": 1.0,
#         "mta_tax": 0.5,
#         "tolls_amount": 0.0,
#         "improvement_surcharge": 1.0,
#         "congestion_surcharge": 2.5,
#         "Airport_fee": 0.0,
#         "trip_duration_minutes": 15.0,
#         "trip_speed_mph": 12.0,
#         "log_trip_distance": 1.2,
#         "fare_per_mile": 6.5,
#         "fare_per_minute": 1.33,
#         "pickup_borough_Bronx": 0,
#         "pickup_borough_Brooklyn": 0,
#         "pickup_borough_EWR": 0,
#         "pickup_borough_Manhattan": 1,
#         "pickup_borough_N/A": 0,
#         "pickup_borough_Queens": 0,
#         "pickup_borough_Staten Island": 0,
#         "pickup_borough_Unknown": 0,
#         "dropoff_borough_Bronx": 0,
#         "dropoff_borough_Brooklyn": 0,
#         "dropoff_borough_EWR": 0,
#         "dropoff_borough_Manhattan": 1,
#         "dropoff_borough_N/A": 0,
#         "dropoff_borough_Queens": 0,
#         "dropoff_borough_Staten Island": 0,
#         "dropoff_borough_Unknown": 0
#     }
#     response = client.post("/predict", json=wrong_type_payload)
#     assert response.status_code == 422

# def test_wrong_data_types_for_batch_prediction(client):
#     wrong_type_batch_payload = {
#         "records": [
#             {
#                 "payment_type": "one", # Should be an integer
#                 "fare_amount": 20.0,
#                 "extra": 1.0,
#                 "mta_tax": 0.5,
#                 "tolls_amount": 0.0,
#                 "improvement_surcharge": 1.0,
#                 "congestion_surcharge": 2.5,
#                 "Airport_fee": 0.0,
#                 "trip_duration_minutes": 15.0,
#                 "trip_speed_mph": 12.0,
#                 "log_trip_distance": 1.2,
#                 "fare_per_mile": 6.5,
#                 "fare_per_minute": 1.33,
#                 "pickup_borough_Bronx": 0,
#                 "pickup_borough_Brooklyn": 0,
#                 "pickup_borough_EWR": 0,
#                 "pickup_borough_Manhattan": 1,
#                 "pickup_borough_N/A": 0,
#                 "pickup_borough_Queens": 0,
#                 "pickup_borough_Staten Island": 0,
#                 "pickup_borough_Unknown": 0,
#                 "dropoff_borough_Bronx": 0,
#                 "dropoff_borough_Brooklyn": 0,
#                 "dropoff_borough_EWR": 0,
#                 "dropoff_borough_Manhattan": 1,
#                 "dropoff_borough_N/A": 0,
#                 "dropoff_borough_Queens": 0,
#                 "dropoff_borough_Staten Island": 0,
#                 "dropoff_borough_Unknown": 0
#             },
#             {
#                 "payment_type": "one", # Should be an integer
#                 "fare_amount": 10.0,
#                 "extra": 0.5,
#                 "mta_tax": 0.5,
#                 "tolls_amount": 0.0,
#                 "improvement_surcharge": 1.0,
#                 "congestion_surcharge": 2.5,
#                 "Airport_fee": 0.0,
#                 "trip_duration_minutes": 8.0,
#                 "trip_speed_mph": 10.0,
#                 "log_trip_distance": 0.8,
#                 "fare_per_mile": 5.0,
#                 "fare_per_minute": 1.25,
#                 "pickup_borough_Bronx": 0,
#                 "pickup_borough_Brooklyn": 1,
#                 "pickup_borough_EWR": 0,
#                 "pickup_borough_Manhattan": 0,
#                 "pickup_borough_N/A": 0,
#                 "pickup_borough_Queens": 0,
#                 "pickup_borough_Staten Island": 0,
#                 "pickup_borough_Unknown": 0,
#                 "dropoff_borough_Bronx": 0,
#                 "dropoff_borough_Brooklyn": 1,
#                 "dropoff_borough_EWR": 0,
#                 "dropoff_borough_Manhattan": 0,
#                 "dropoff_borough_N/A": 0,
#                 "dropoff_borough_Queens": 0,
#                 "dropoff_borough_Staten Island": 0,
#                 "dropoff_borough_Unknown": 0
#             }
#         ]
#     }

#     response = client.post("/predict/batch", json=wrong_type_batch_payload)
#     assert response.status_code == 422
