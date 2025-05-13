FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
# ENV MLFLOW_TRACKING_URI (Set at runtime by Kubernetes is preferred)

# Copy requirements to /app first (one level up from WORKDIR)
COPY src/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory content into the WORKDIR (/app/src)
COPY src ./src

EXPOSE 8000

# Command for development with live reload (relative to WORKDIR /app/src):
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src/api"]

# Commented out old commands
# CMD ["python", "./src/api/main.py"]
# CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 