import mlflow
import os
import sys # Import sys for exit codes

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
# EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'HealthPredict_Training_HPO_Airflow') # Not directly used in this script but good for context
MODEL_NAMES = ["HealthPredict_LogisticRegression", "HealthPredict_RandomForest", "HealthPredict_XGBoost"]

print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

all_checks_passed = True

for model_name in MODEL_NAMES:
    print(f"--- Verifying Model: {model_name} ---")
    try:
        registered_model = client.get_registered_model(model_name)
        # get_registered_model raises mlflow.exceptions.RestException if not found, 
        # so an explicit check for None might not be necessary if we let the exception be caught.
        # However, for clarity if the API changes or for other clients, explicit checks are safer.
        if not registered_model: # Defensive check
            print(f"ERROR: Registered model '{model_name}' not found (get_registered_model returned None/False).")
            all_checks_passed = False
            continue
        
        print(f"Found registered model: '{model_name}'.")
        
        latest_versions = registered_model.latest_versions
        if not latest_versions:
            print(f"ERROR: No versions found for model '{model_name}'.")
            all_checks_passed = False
            continue

        production_version_found = False
        # Check the most recent version among latest_versions for 'Production' stage
        # MLflow's latest_versions list is already sorted with the most recent (highest version number) first.
        version_info = latest_versions[0] # The actual latest version
        if version_info.current_stage == 'Production':
            production_version_found = True
            print(f"  Latest Version {version_info.version} is in 'Production' stage.")
            print(f"    Source run_id: {version_info.run_id}")
            
            # Verify source run details
            try:
                source_run = mlflow.get_run(version_info.run_id)
                expected_model_type_tag = model_name.replace('HealthPredict_', '')
                
                if source_run.data.tags.get('best_hpo_model') == 'True' and \
                   source_run.data.tags.get('model_name') == expected_model_type_tag:
                    print(f"    Source run ID {version_info.run_id} tags verified (best_hpo_model='True', model_name='{expected_model_type_tag}').")
                else:
                    print(f"ERROR: Source run ID {version_info.run_id} tags not as expected for {model_name}.")
                    print(f"      Expected: best_hpo_model='True', model_name='{expected_model_type_tag}'")
                    print(f"      Found Tags: {source_run.data.tags}")
                    all_checks_passed = False
            except Exception as e_run:
                print(f"ERROR: Could not fetch or verify source run {version_info.run_id} for {model_name}: {e_run}")
                all_checks_passed = False
        
        if not production_version_found:
            print(f"ERROR: Latest version ({latest_versions[0].version if latest_versions else 'N/A'}) is not in 'Production' stage for model '{model_name}'. Current stage: {latest_versions[0].current_stage if latest_versions else 'N/A'}")
            all_checks_passed = False

    except mlflow.exceptions.RestException as e_rest:
        if "RESOURCE_DOES_NOT_EXIST" in str(e_rest) or e_rest.get_http_status_code() == 404:
            print(f"ERROR: Registered model '{model_name}' not found (MLflow API 404/ResourceDoesNotExist).")
        else:
            print(f"ERROR: MLflow API RestException while verifying model '{model_name}': {e_rest}")
        all_checks_passed = False
    except Exception as e_general:
        print(f"ERROR: A general exception occurred while verifying model '{model_name}': {e_general}")
        all_checks_passed = False
    print("-" * 40)

if all_checks_passed:
    print("\\nSUCCESS: All MLflow registration and staging checks passed.")
    sys.exit(0)
else:
    print("\\nFAILURE: Some MLflow registration or staging checks failed.")
    sys.exit(1) 