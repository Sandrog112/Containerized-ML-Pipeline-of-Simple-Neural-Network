# Start from a base Python image
FROM python:3.10

# Initialize model and settings with default values
ARG model_name=iris_classifier.pth
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY data_process/data/ /app/data_process/data

# Copy the specific model to Docker 
COPY models/${model_name} /app/models/${model_name}

# Copy the inference code
COPY inference /app/inference

# Copy utility scripts and configurations
COPY ${CONF_PATH} /app

# Install required packages
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# Run the inference script 
CMD ["python3", "inference/run.py"]

