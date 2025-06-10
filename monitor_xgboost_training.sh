#!/bin/bash

echo "=== PRODUCTION XGBOOST TRAINING MONITOR ==="
echo "Monitoring DAG: manual__2025-06-10T00:32:44+00:00"
echo "Training task: run_training_and_hpo"
echo "Started: 2025-06-10T00:33:05+00:00"
echo ""

check_count=0
while true; do
    check_count=$((check_count + 1))
    current_time=$(date)
    echo "=== CHECK #$check_count at $current_time ==="
    
    # Check DAG overall status
    dag_status=$(docker compose --env-file .env -f mlops-services/docker-compose.yml exec airflow-scheduler airflow dags list-runs -d health_predict_continuous_improvement | head -3 | tail -1 | awk '{print $3}')
    echo "DAG Status: $dag_status"
    
    if [[ "$dag_status" == "success" ]]; then
        echo "🎉 DAG COMPLETED SUCCESSFULLY!"
        break
    elif [[ "$dag_status" == "failed" ]]; then
        echo "❌ DAG FAILED - Need to investigate"
        break
    elif [[ "$dag_status" == "running" ]]; then
        echo "🔄 DAG still running - checking task status..."
        
        # Check which task is currently running
        current_task=$(docker compose --env-file .env -f mlops-services/docker-compose.yml exec airflow-scheduler airflow tasks states-for-dag-run health_predict_continuous_improvement manual__2025-06-10T00:32:44+00:00 | grep "running" | awk '{print $3}')
        echo "Current task: $current_task"
        
        # Calculate runtime
        start_time=$(date -d '2025-06-10T00:33:05+00:00' +%s 2>/dev/null || echo $(date +%s))
        current_timestamp=$(date +%s)
        runtime_seconds=$((current_timestamp - start_time))
        runtime_minutes=$((runtime_seconds / 60))
        echo "Training runtime: ${runtime_minutes} minutes (${runtime_seconds} seconds)"
        
        echo "Next check in 3 minutes..."
        sleep 180  # Check every 3 minutes
    else
        echo "Unknown status: $dag_status"
        sleep 60
    fi
done

echo "Monitoring complete at $(date)" 