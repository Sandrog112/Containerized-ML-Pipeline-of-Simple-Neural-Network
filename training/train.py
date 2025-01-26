import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from time import time
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Defining directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../models'))
TRAIN_PATH = os.path.abspath(os.path.join(ROOT_DIR, '../data_process/data/train_data.csv'))

# Create models directory if it does not exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Function to load and process data
def load_data(train_path):
    try:
        df = pd.read_csv(train_path) 
    except FileNotFoundError:
        logger.error(f"Error: The file at {train_path} was not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        sys.exit(1)

    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    # Standardize features and split data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Log dataset sizes
    logger.info(f"Training dataset size: {X_train.shape[0]} rows, {X_train.shape[1]} columns.")
    logger.info(f"Test dataset size: {X_test.shape[0]} rows, {X_test.shape[1]} columns.")
    
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, X_test, y_train, y_test

# Neural network model for Iris dataset classification
class IrisClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training function for the model
def train_model(model, criterion, optimizer, train_loader, num_epochs=50):
    start_time = time()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward() 
            optimizer.step() 
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    training_time = time() - start_time
    logger.info(f"Total training time: {training_time:.4f} seconds")

# Evaluation function to calculate accuracy and log results
def evaluate_model(model, X_test, y_test):
    start_time = time()
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / len(y_test)
    inference_time = time() - start_time
    logger.info(f"Inference time: {inference_time:.4f} seconds")
    logger.info(f"Model accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # Load and prepare the data
    X_train, X_test, y_train, y_test = load_data(TRAIN_PATH)
    
    # Prepare data for DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    input_dim = X_train.shape[1]
    hidden_dim = 16
    output_dim = len(torch.unique(y_train))
    
    model = IrisClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    logger.info("Training the model...")
    train_model(model, criterion, optimizer, train_loader)
    
    # Save the trained model
    model_save_path = os.path.join(MODELS_DIR, 'iris_classifier.pth')
    try:
        torch.save(model.state_dict(), model_save_path)
        logger.info(f'Model saved to {model_save_path}')
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        sys.exit(1)
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    accuracy = evaluate_model(model, X_test, y_test)
    logger.info(f"Final model accuracy on test data: {accuracy:.4f}")
    
    logger.info("Script completed successfully.")
