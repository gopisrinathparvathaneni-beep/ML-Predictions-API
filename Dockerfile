# Dockerfile

# Stage 1: Build environment for installing dependencies
FROM python:3.9-slim-buster as builder

# Set working directory inside the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Minimal runtime environment
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# If using FastAPI/Uvicorn, copy the uvicorn executable
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application source code and model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Expose the port where your API service will listen
EXPOSE 8000

# Define environment variables (e.g., default model path)
ENV MODEL_PATH=/app/models/my_classifier_model.h5
ENV LOG_LEVEL=INFO

# Command to run the application (FastAPI example using Uvicorn)
# The `--host 0.0.0.0` ensures the server is accessible from outside the container
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]