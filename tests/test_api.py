import os
import sys

# Ensure project root is on sys.path so `src` package can be imported during tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["SKIP_MODEL_LOAD"] = "1"

from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

import pytest
from unittest.mock import patch
import io
from PIL import Image
import numpy as np

# `client` is created above using httpx.ASGITransport

def test_health_check_endpoint():
    # Test the /health endpoint to ensure it returns a 200 OK status
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is healthy and model is loaded."}

# Use patches to mock the heavy ML model operations during unit tests
@patch('src.main.load_model')
@patch('src.main.predict_image')
@patch('src.main.preprocess_image')
def test_predict_success_with_mocked_model(mock_preprocess_image, mock_predict_image, mock_load_model):
    # Configure mocks to return predefined values for controlled testing
    mock_model = mock_load_model.return_value
    mock_model.predict.return_value = np.array([[0.05, 0.05, 0.05, 0.05, 0.05, 0.75, 0.0, 0.0, 0.0, 0.0]])
    mock_preprocess_image.return_value = "mock_preprocessed_image_array"  # Return a dummy preprocessed image
    mock_predict_image.return_value = {"class_label": "5", "probabilities": [0.05, 0.05, 0.05, 0.05, 0.05, 0.75, 0.0, 0.0, 0.0, 0.0]}  # Mock prediction result for digit 5

    # Create a dummy image file in-memory for the test request
    dummy_image = Image.new('L', (28, 28), color=128)  # Grayscale image
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Reset stream position to the beginning

    # Send a POST request to the /predict endpoint
    response = client.post(
        "/predict",
        files={
            "file": ("test_image.png", img_byte_arr, "image/png")
        }
    )
    # Assert the response status code and content
    assert response.status_code == 200
    assert response.json() == {"class_label": "5", "probabilities": [0.05, 0.05, 0.05, 0.05, 0.05, 0.75, 0.0, 0.0, 0.0, 0.0]}
    # Verify that the mocked functions were called as expected
    mock_preprocess_image.assert_called_once()
    mock_predict_image.assert_called_once()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the ML API" in response.json()["message"]

def test_predict_invalid_file_type_handling():
    # Test behavior when an unsupported file type is sent
    response = client.post(
        "/predict",
        files={
            "file": ("document.txt", b"This is not an image.", "text/plain")
        }
    )
    # Assert that the API correctly rejects invalid file types with a 400 status
    assert response.status_code == 400
    assert "Only image files (e.g., JPEG, PNG) are allowed for prediction." in response.json()["detail"]

def test_predict_missing_file_upload():
    # Test behavior when no file is uploaded
    response = client.post(
        "/predict",
        data={}
    )
    # FastAPI automatically handles missing required fields with a 422 Unprocessable Entity
    assert response.status_code == 422
    assert "Field required" in response.json()["detail"][0]["msg"]

@patch('src.main.preprocess_image')
def test_predict_preprocessing_error_handling(mock_preprocess_image):
    mock_preprocess_image.side_effect = ValueError("Corrupted image data")
    dummy_image = Image.new('L', (28, 28), color=128)
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": ("corrupt_image.png", img_byte_arr, "image/png")
        }
    )
    assert response.status_code == 422
    assert "Image processing failed" in response.json()["detail"]

