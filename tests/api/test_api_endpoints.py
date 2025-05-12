import pytest
import requests
import os

# Get the API base URL from environment variable or use default
API_BASE_URL = os.getenv("API_BASE_URL", "http://192.168.49.2:30854")

def test_health_check():
    """
    Tests the /health endpoint to ensure the API is running and the model is loaded.
    """
    health_url = f"{API_BASE_URL}/health"
    try:
        response = requests.get(health_url, timeout=10) # Added timeout

        # Check if the request was successful
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        # Check status code
        assert response.status_code == 200

        # Check response body
        response_json = response.json()
        assert response_json.get("status") == "ok"
        # Check if model status is reported as loaded (based on API implementation)
        assert response_json.get("model_status") == "loaded"
        print(f"Health check passed: {response_json}")

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request to {health_url} failed: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during health check: {e}")

# Add more tests below for the /predict endpoint 