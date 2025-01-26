import os
import sys
import pandas as pd
import logging
import json
import numpy as np
from sklearn import datasets
from time import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.getenv('CONF_PATH', 'settings.json')

# Check if configuration file exists
if not os.path.exists(CONF_FILE):
    logger.error(f"Configuration file '{CONF_FILE}' not found!")
    sys.exit(1)

logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defining data and table paths
logger.info("Defining paths...")
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, conf['general']['data_dir']))
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

# Function to get project directory
def get_project_dir(data_dir):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), data_dir))

# Configure logging function
def configure_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

# IrisSetGenerator class to generate and save Iris dataset
@singleton
class IrisSetGenerator:
    def __init__(self):
        self.df = None

    def create(self, is_labeled: bool = True, save_path: os.path = None):
        logger.info("Loading Iris dataset...")
        
        try:
            iris = datasets.load_iris() 
        except Exception as e:
            logger.error(f"Failed to load Iris dataset: {e}")
            sys.exit(1)
        
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.df['target'] = iris.target
        
        if not is_labeled:
            self.df = self.df.drop(columns=['target'])
        
        if save_path:
            self.save(self.df, save_path)
        
        return self.df

    def save(self, df: pd.DataFrame, out_path: os.path):
        logger.info(f"Saving data to {out_path}...")
        try:
            df.to_csv(out_path, index=False)  
        except Exception as e:
            logger.error(f"Failed to save data to {out_path}: {e}")
            sys.exit(1)

# Main function
if __name__ == "__main__":
    configure_logging()  
    logger.info("Starting script...")

    # Generate and save labeled and unlabeled Iris datasets
    gen = IrisSetGenerator()
    train_df = gen.create(is_labeled=True, save_path=TRAIN_PATH)
    inference_df = gen.create(is_labeled=False, save_path=INFERENCE_PATH)
    
    # Log dataset sizes
    logger.info(f"Training dataset size: {train_df.shape[0]} rows, {train_df.shape[1]} columns.")
    logger.info(f"Inference dataset size: {inference_df.shape[0]} rows, {inference_df.shape[1]} columns.")

    # Train a simple model and log time and quality
    logger.info("Training model...")

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    start_time = time()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    training_time = time() - start_time
    logger.info(f"Training time: {training_time:.4f} seconds.")
    
    # Make predictions and log quality (accuracy)
    logger.info("Making predictions...")

    start_time = time()
    y_pred = model.predict(X_train)
    inference_time = time() - start_time
    logger.info(f"Inference time: {inference_time:.4f} seconds.")
    
    accuracy = accuracy_score(y_train, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    logger.info("Script completed successfully.")
