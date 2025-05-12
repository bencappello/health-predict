FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
# ENV MLFLOW_TRACKING_URI (Set at runtime by Kubernetes is preferred)

# Copy requirements first to leverage Docker cache
COPY src/api/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory into the /app/src directory in the image
COPY src ./src

EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# This assumes main.py is in /app/src/api/main.py
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 