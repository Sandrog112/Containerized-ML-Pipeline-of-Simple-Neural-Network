import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import logging
import time  

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train import IrisClassifier

# Define directories and paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../models'))
INFERENCE_PATH = os.path.abspath(os.path.join(ROOT_DIR, '../data_process/data/inference_data.csv'))
MODEL_PATH = os.path.join(MODELS_DIR, 'iris_classifier.pth')

# Creating the function to load inference data
def load_inference_data(inference_path):
    try:
        df = pd.read_csv(inference_path) 
        logger.info(f"Inference data loaded successfully from {inference_path}.")
    except FileNotFoundError:
        logger.error(f"Error: The file at {inference_path} was not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        sys.exit(1)

    # Standardizing the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor

# Function to run inference using the trained model
def run_inference(model, data_loader):
    model.eval()  
    predictions = []
    start_time = time.time() 
    with torch.no_grad():  
        for inputs in data_loader:
            outputs = model(inputs[0])  
            _, predicted = torch.max(outputs, 1)  
            predictions.extend(predicted.numpy())
    
    inference_time = time.time() - start_time  

    if not predictions:
        logger.error("Error: No predictions were made. Please check the model and data.")
        sys.exit(1)

    logger.info(f"Inference completed, {len(predictions)} predictions made.")
    logger.info(f"Inference time: {inference_time:.4f} seconds.") 
    return predictions

if __name__ == "__main__":
    # Checking if the model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Error: The model file at {MODEL_PATH} does not exist.")
        sys.exit(1)
    
    # Define model dimensions
    input_dim = 4
    hidden_dim = 16
    output_dim = 3

    # Initialize model and load trained weights
    model = IrisClassifier(input_dim, hidden_dim, output_dim)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))  
        logger.info(f"Model loaded successfully from {MODEL_PATH}.")
    except RuntimeError as e:
        logger.error(f"Error loading the model: {e}")
        sys.exit(1)

    # Load inference data
    X_infer = load_inference_data(INFERENCE_PATH)
    infer_dataset = TensorDataset(X_infer)
    infer_loader = DataLoader(infer_dataset, batch_size=16, shuffle=False)

    predictions = run_inference(model, infer_loader)
    
    # Define the output path and save predictions to a CSV file
    output_path = os.path.join(ROOT_DIR, '../data_process/data/inference_results.csv')
    try:
        pd.DataFrame(predictions, columns=['predictions']).to_csv(output_path, index=False)
        logger.info(f'Inference results saved to {output_path}')
    except Exception as e:
        logger.error(f"Error saving the inference results: {e}")
        sys.exit(1)

    logger.info("Inference process completed successfully.")
