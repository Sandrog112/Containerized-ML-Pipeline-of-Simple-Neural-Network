# Use a base Python image
FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set the working directory in the container
WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy the training code
COPY training /app/training

# Copy the settings file (could be overridden by providing a different config file)
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# Run the training script
CMD ["python3", "training/train.py"]