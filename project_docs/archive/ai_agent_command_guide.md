# AI Agent Command Guide & Best Practices

This document outlines common pitfalls, effective workarounds, and best practices for AI coding assistants working on the Health Predict MLOps project. Its purpose is to streamline troubleshooting and ensure consistent command usage.

## 1. Reading Airflow Task Logs (and other problematic files)

**Problem:**
The `read_file` tool has shown inconsistent behavior when attempting to read Airflow task log files located within `mlops-services/logs/`. While `list_dir` can see these files and their sizes, `read_file` often fails, reporting that the file cannot be found. This issue is particularly prevalent for files on Docker volume mounts or files recently written by processes within Docker containers (like Airflow workers).

**Hypotheses for `read_file` tool failure:**
*   **Path interpretation/resolution:** The tool might have subtle differences in handling paths for volume-mounted files compared to standard file system paths.
*   **Internal limitations:** The tool could have restrictions related to file types, specific file systems (especially mounted ones), or concurrent access/locking issues with files actively being written to.

**Solution/Workaround:**
Use `run_terminal_cmd` with the `cat` command to read these files. This method has proven reliable as it leverages the host system's direct file access capabilities as the `ubuntu` user.

**Correct Command Format:**
To inspect an Airflow task log (or any file `read_file` struggles with):
```bash
cat <path_to_file>
```

**Example:**
To read the log for the `update_kubernetes_deployment` task from DAG run `manual__2025-05-14T23:02:05+00:00` of the `health_predict_api_deployment` DAG:
```bash
cat mlops-services/logs/dag_id=health_predict_api_deployment/run_id=manual__2025-05-14T23:02:05+00:00/task_id=update_kubernetes_deployment/attempt=1.log
```

**Key Takeaway:**
If `read_file` fails for a log file (especially Airflow logs), default to using `run_terminal_cmd` with `cat` for reliable file content retrieval. This significantly improves the debugging feedback loop.

## 2. (Future Learnings Placeholder)

*(This section will be updated as new common issues and solutions are identified.)*

## 3. Checking Airflow DAG Run Status

**Problem:**
When checking the status of Airflow DAG runs using `airflow dags list-runs` via `docker-compose exec -T airflow-scheduler ...`, several issues were encountered with command arguments:
*   Invalid output format: Supplying `-o csv` resulted in an error, as `csv` is not a supported output format for this command in the Airflow version being used.
*   Unrecognized arguments: Using `--no-header` also caused an error as it's not a valid option.

**Solution/Workaround:**
To reliably check DAG run status and retrieve information for specific runs:
1.  Use a supported output format, such as `plain`, `json`, or `yaml`. The `plain` format is simple for direct `grep` or visual inspection. `json` or `yaml` can be more robust if programmatic parsing of the full output is needed.
2.  Omit the `--no-header` argument. If using `plain` output, be prepared to handle or ignore the header line during parsing if necessary.
3.  Use `grep` (or similar tools) to filter the output for the specific `run_id` if you only need information about a particular run.

**Correct Command Format (Example with `plain` output and `grep`):**
```bash
docker-compose -f mlops-services/docker-compose.yml exec -T airflow-scheduler airflow dags list-runs --dag-id <your_dag_id> -o plain | grep '<your_run_id>'
```

**Example:**
To check the status of DAG run `manual__2025-05-15T00:52:47+00:00` for the `health_predict_training_hpo` DAG:
```bash
docker-compose -f mlops-services/docker-compose.yml exec -T airflow-scheduler airflow dags list-runs --dag-id health_predict_training_hpo -o plain | grep 'manual__2025-05-15T00:52:47+00:00'
```
This will output the line(s) corresponding to that specific DAG run, showing its current state.

**Alternative (getting all runs in JSON and parsing later if needed):**
```bash
docker-compose -f mlops-services/docker-compose.yml exec -T airflow-scheduler airflow dags list-runs --dag-id <your_dag_id> -o json
```

**Key Takeaway:**
Always verify supported arguments and output formats for Airflow CLI commands, especially when they are executed through `docker-compose exec`. When in doubt, use `airflow dags list-runs --help` within the container to see available options. For targeted information retrieval from `plain` output, `grep` is effective. 