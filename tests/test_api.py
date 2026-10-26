#import test client so we can send simulated http requestS DIRECTLY TO FastAPI app without having to run the server
from fastapi.testclient import TestClient
#import the application instance from main.py
from main import app
#create a reuable test client for all test functions
client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
def test_valid_iris_prediction():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "species" in result
    assert result["species"] in ["setosa", "versicolor", "virginica"]
    assert result["prediction"] in [0, 1, 2]
def test_invalid_iris_prediction():
    payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity due to validation error
    