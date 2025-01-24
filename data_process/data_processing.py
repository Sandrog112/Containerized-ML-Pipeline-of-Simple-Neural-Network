import os
import sys
import pandas as pd
import logging
import json
from sklearn import datasets

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

CONF_FILE = os.getenv('CONF_PATH', 'settings.json')

logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

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

def get_project_dir(data_dir):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), data_dir))

def configure_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

@singleton
class IrisSetGenerator():
    def __init__(self):
        self.df = None

    def create(self, is_labeled: bool = True, save_path: os.path = None):
        logger.info("Loading Iris dataset...")
        iris = datasets.load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.df['target'] = iris.target
        
        if not is_labeled:
            self.df = self.df.drop(columns=['target'])
        
        if save_path:
            self.save(self.df, save_path)
        return self.df

    def save(self, df: pd.DataFrame, out_path: os.path):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)

if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    gen = IrisSetGenerator()
    gen.create(is_labeled=True, save_path=TRAIN_PATH)
    gen.create(is_labeled=False, save_path=INFERENCE_PATH)
    logger.info("Script completed successfully.")
