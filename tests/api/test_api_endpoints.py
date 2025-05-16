import pytest
print("[DEBUG] Pytest imported")
import requests
print("[DEBUG] Requests imported")
import os
print("[DEBUG] OS imported")
import copy # For deep copying payloads
print("[DEBUG] Copy imported")

# Determine API base URL from environment variables
# These will be set by the Airflow DAG (construct_test_command task)
MINIKUBE_IP = os.getenv("MINIKUBE_IP", "127.0.0.1") # Default for local manual runs
K8S_NODE_PORT = os.getenv("K8S_NODE_PORT") # Default for local manual runs

API_CLUSTER_IP = os.getenv("API_CLUSTER_IP")
API_SERVICE_PORT = os.getenv("API_SERVICE_PORT")

if API_CLUSTER_IP and API_SERVICE_PORT:
    API_BASE_URL = f"http://{API_CLUSTER_IP}:{API_SERVICE_PORT}"
elif MINIKUBE_IP and K8S_NODE_PORT: # Fallback for older setup or manual runs
    API_BASE_URL = f"http://{MINIKUBE_IP}:{K8S_NODE_PORT}"
else:
    # If neither ClusterIP/Port nor NodePort/IP are available, tests cannot run.
    # Pytest will ideally be skipped or this will cause a failure at session scope.
    API_BASE_URL = "http://ERROR_UNDETERMINED_API_URL:0"

print(f"[DEBUG] Test API Base URL determined: {API_BASE_URL}")

print("[DEBUG] Attempting to instantiate requests.Session()")
api_session = None # Initialize
try:
    api_session = requests.Session()
    print("[DEBUG] requests.Session() instantiated successfully.")
    api_session.trust_env = False # To bypass potential proxy issues
    print("[DEBUG] api_session.trust_env set to False.")
    api_session.timeout = (5, 7) # Connect timeout = 5s, Read timeout = 7s
    print("[DEBUG] api_session.timeout set to (5, 7).")
except Exception as e:
    print(f"[DEBUG] ERROR instantiating or configuring requests.Session(): {e}")
    # Optionally, re-raise or handle if tests cannot run without a session

# Define a sample valid payload based on the Pydantic model
# Matches the structure expected by the /predict endpoint and feature engineering
# Ensure field names match Pydantic model aliases (e.g., 'race' for 'race')
VALID_PAYLOAD = {
    "patient_nbr": 135,
    "gender": "Female",
    "age": "[70-80)",
    "admission-type-id": 1,
    "discharge-disposition-id": 1,
    "admission-source-id": 7,
    "time-in-hospital": 5,
    "num-lab-procedures": 40,
    "num-procedures": 1,
    "num-medications": 15,
    "number-outpatient": 0,
    "number-emergency": 0,
    "number-inpatient": 0,
    "diag_1": "250.8",  # Example diagnosis code (ensure format is correct)
    "diag_2": "428",    # Example diagnosis code
    "diag_3": "401",    # Example diagnosis code
    "number-diagnoses": 9,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "No",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "No"
    # Note: Removed 'race' as it wasn't in the missing fields list. Add if needed.
    # Add other fields if the model expects more than these.
}

@pytest.fixture(scope="module")
def service_url():
    # When run from the DAG, API_CLUSTER_IP and API_SERVICE_PORT must be set.
    # The K8S_NODE_PORT check is a fallback for old manual runs, but for DAG execution,
    # the new variables are primary.
    if API_CLUSTER_IP and API_SERVICE_PORT:
        # This confirms the primary variables used for API_BASE_URL are available.
        pass # Expected path when run from DAG
    elif K8S_NODE_PORT: 
        # This is a fallback for local runs where only K8S_NODE_PORT might be set
        pass # Fallback path for local/manual runs
    else:
        pytest.fail("Neither API_CLUSTER_IP/API_SERVICE_PORT nor K8S_NODE_PORT environment variables are set. Cannot determine service URL.")
    return API_BASE_URL

# --- Health Check Test ---

def test_health_check(service_url):
    """
    Tests the /health endpoint to ensure the API is running and the model is loaded.
    """
    health_url = f"{service_url}/health"
    print(f"Testing health endpoint: {health_url}")
    try:
        response = api_session.get(health_url)
        response.raise_for_status()
        assert response.status_code == 200
        response_json = response.json()
        assert response_json.get("status") == "healthy"
        # Check that the message indicates model is loaded
        assert response_json.get("model_loaded") is True
        print(f"Health check passed: {response_json}")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Health check request failed: {e}")

# --- Predict Endpoint Tests ---

def test_predict_valid_input(service_url):
    """Tests the /predict endpoint with a valid input payload."""
    predict_url = f"{service_url}/predict"
    print(f"Testing predict endpoint with valid data: {predict_url}")
    try:
        response = api_session.post(predict_url, json=VALID_PAYLOAD)
        response.raise_for_status()
        assert response.status_code == 200
        response_json = response.json()
        assert "prediction" in response_json
        assert "probability_score" in response_json
        assert response_json["prediction"] in [0, 1]
        assert 0.0 <= response_json["probability_score"] <= 1.0
        print(f"Predict valid input test passed: {response_json}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            print(f"Received 422 Error. Response body: {e.response.text}")
        pytest.fail(f"Request to {predict_url} with valid input failed: {e}")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request to {predict_url} with valid input failed: {e}")

def test_predict_missing_field(service_url):
    """Tests the /predict endpoint with a payload missing a required field."""
    predict_url = f"{service_url}/predict"
    print(f"Testing predict endpoint with missing field: {predict_url}")
    payload = copy.deepcopy(VALID_PAYLOAD)
    # Assuming 'time-in-hospital' is a required field. Adjust if necessary.
    if "time-in-hospital" in payload:
        del payload["time-in-hospital"]
    else:
        pytest.skip("Skipping missing field test: 'time-in-hospital' not in sample payload for deletion.")

    try:
        response = api_session.post(predict_url, json=payload)
        assert response.status_code == 422 # FastAPI/Pydantic validation error
        print(f"Predict missing field test passed with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        # A 422 is an expected "bad" response from client side, not a server/network error for the test to fail on
        if e.response is not None and e.response.status_code == 422:
            assert e.response.status_code == 422
            print(f"Predict missing field test passed with status {e.response.status_code}")
        else:
            pytest.fail(f"Request to {predict_url} with missing field failed unexpectedly: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during predict missing field test: {e}")

def test_predict_invalid_data_type(service_url):
    """Tests the /predict endpoint with a payload having an incorrect data type for a field."""
    predict_url = f"{service_url}/predict"
    print(f"Testing predict endpoint with invalid data type: {predict_url}")
    payload = copy.deepcopy(VALID_PAYLOAD)
    # Assuming 'time-in-hospital' should be an int. Send as string.
    payload["time-in-hospital"] = "five_days" # Invalid type

    try:
        response = api_session.post(predict_url, json=payload)
        assert response.status_code == 422 # FastAPI/Pydantic validation error
        print(f"Predict invalid data type test passed with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 422:
            assert e.response.status_code == 422
            print(f"Predict invalid data type test passed with status {e.response.status_code}")
        else:
            pytest.fail(f"Request to {predict_url} with invalid data type failed unexpectedly: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during predict invalid data type test: {e}")

def test_predict_malformed_json(service_url):
    """Test /predict with malformed JSON (though requests might catch this earlier)."""
    malformed_json_string = '{"patient_nbr": 1, "race": "Caucasian", ...this is not valid json'
    response = api_session.post(
        f"{service_url}/predict", 
        data=malformed_json_string, 
        headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 400 # Bad Request for malformed JSON

# Consider adding more tests for other edge cases or specific feature interactions
# Example: Test with values that might cause issues in feature engineering

def test_predict_edge_case_age(service_url):
    """Test /predict with an edge case for the 'age' field."""
    payload_edge_age = copy.deepcopy(VALID_PAYLOAD)
    payload_edge_age["age"] = "[0-10)" # Youngest age group
    response = api_session.post(f"{service_url}/predict", json=payload_edge_age)
    assert response.status_code == 200 
    # Add more assertions based on expected behavior for this age group

def test_predict_unknown_race(service_url):
    """Test /predict with an unknown 'race' to see how OHE handles it (if configured to)."""
    payload_unknown_race = copy.deepcopy(VALID_PAYLOAD)
    payload_unknown_race["race"] = "Martian" # An unknown race
    response = api_session.post(f"{service_url}/predict", json=payload_unknown_race)
    # The behavior here depends on how your preprocessor's OneHotEncoder handles unknown values.
    # If it's set to 'ignore', it should still produce a 200. If it errors, it might be 422 or 500.
    # Assuming it's configured to ignore or handle gracefully:
    assert response.status_code == 200 
    # Add assertions based on expected output for unknown race 