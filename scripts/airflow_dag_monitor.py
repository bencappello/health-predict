#!/usr/bin/env python3

import argparse
import subprocess
import json
import os
import sys
import time
from datetime import datetime

def run_command(command):
    """Run a shell command and return the output."""
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        shell=True,
        text=True
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Error message: {stderr}")
        return None
    return stdout

def check_dag_status(dag_id, run_id=None):
    """Check the status of a specific DAG run."""
    if run_id:
        cmd = f"docker-compose exec -T airflow-webserver airflow dags state {dag_id} {run_id}"
        state = run_command(cmd).strip() if run_command(cmd) else "unknown"
        
        print(f"\n=== DAG RUN STATUS FOR {dag_id} (RUN_ID: {run_id}) ===")
        print(f"State: {state}")
        return state
    else:
        cmd = f"docker-compose exec -T airflow-webserver airflow dags list-runs -d {dag_id} --output json"
        output = run_command(cmd)
        if not output:
            return None
        
        try:
            runs = json.loads(output)
            if runs:
                latest_run = runs[0]  # Most recent run
                run_id = latest_run.get('run_id')
                state = latest_run.get('state')
                start_date = latest_run.get('start_date')
                
                print(f"\n=== LATEST DAG RUN FOR {dag_id} ===")
                print(f"Run ID: {run_id}")
                print(f"State: {state}")
                print(f"Start Date: {start_date}")
                
                return state
            else:
                print(f"No runs found for DAG: {dag_id}")
                return None
        except json.JSONDecodeError:
            print(f"Could not parse JSON output from Airflow CLI: {output}")
            return None

def check_task_statuses(dag_id, run_id):
    """Check the status of all tasks for a specific DAG run."""
    cmd = f"docker-compose exec -T airflow-webserver airflow tasks list {dag_id} --tree"
    task_tree = run_command(cmd)
    if not task_tree:
        return
    
    tasks = []
    for line in task_tree.splitlines():
        line = line.strip()
        if line and not line.startswith('<'):  # Skip header lines
            # Extract task_id from the tree view
            parts = line.split(')')
            if len(parts) > 1:
                task_id = parts[-1].strip()
            else:
                task_id = line.strip()
            
            # Clean up task_id
            task_id = task_id.replace('(', '').replace(')', '').strip()
            if task_id:
                tasks.append(task_id)
    
    print("\n=== TASK STATUSES ===")
    for task_id in tasks:
        cmd = f"docker-compose exec -T airflow-webserver airflow tasks state {dag_id} {task_id} {run_id}"
        state = run_command(cmd).strip() if run_command(cmd) else "unknown"
        
        # Color coding based on state
        if state == 'success':
            color = '\033[92m'  # Green
        elif state == 'failed':
            color = '\033[91m'  # Red
        elif state == 'running':
            color = '\033[94m'  # Blue
        elif state == 'queued':
            color = '\033[93m'  # Yellow
        else:
            color = '\033[0m'   # Default
            
        reset = '\033[0m'
        print(f"{task_id}: {color}{state}{reset}")
        
        # If task failed, get logs
        if state == 'failed':
            print(f"\n=== LOGS FOR FAILED TASK: {task_id} ===")
            cmd = f"docker-compose exec -T airflow-webserver airflow tasks logs {dag_id} {task_id} {run_id} -n 50"
            logs = run_command(cmd)
            if logs:
                # Show just the error part of the logs
                error_section = ""
                in_error = False
                for line in logs.splitlines():
                    if "ERROR" in line:
                        in_error = True
                    if in_error:
                        error_section += line + "\n"
                
                print(error_section or "No error logs found.")

def monitor_dag(dag_id, run_id=None, interval=10, max_time=3600):
    """Monitor a DAG run until it completes or times out."""
    if not run_id:
        # Get the latest run_id
        cmd = f"docker-compose exec -T airflow-webserver airflow dags list-runs -d {dag_id} --output json"
        output = run_command(cmd)
        if output:
            try:
                runs = json.loads(output)
                if runs:
                    run_id = runs[0].get('run_id')
                    print(f"Monitoring latest run with ID: {run_id}")
                else:
                    print(f"No runs found for DAG: {dag_id}")
                    return
            except json.JSONDecodeError:
                print(f"Could not parse JSON output: {output}")
                return
    
    start_time = time.time()
    while time.time() - start_time < max_time:
        # Clear screen for better readability
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"Monitoring DAG: {dag_id}, Run: {run_id}")
        print(f"Time elapsed: {int(time.time() - start_time)} seconds")
        
        state = check_dag_status(dag_id, run_id)
        check_task_statuses(dag_id, run_id)
        
        if state in ['success', 'failed']:
            print(f"\nDAG run {run_id} has completed with state: {state}")
            break
            
        print(f"\nRefreshing in {interval} seconds...")
        time.sleep(interval)
    else:
        print(f"Monitoring timed out after {max_time} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Airflow DAG execution")
    parser.add_argument("--dag_id", required=True, help="ID of the DAG to monitor")
    parser.add_argument("--run_id", help="Optional run ID of the DAG run to monitor (latest if not specified)")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds (default: 10)")
    parser.add_argument("--max_time", type=int, default=3600, help="Maximum monitoring time in seconds (default: 3600)")
    
    args = parser.parse_args()
    
    monitor_dag(args.dag_id, args.run_id, args.interval, args.max_time) 