import os
import sys
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train import IrisClassifier

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../models'))
INFERENCE_PATH = os.path.abspath(os.path.join(ROOT_DIR, '../data_process/data/inference_data.csv'))
MODEL_PATH = os.path.join(MODELS_DIR, 'iris_classifier.pth')

def load_inference_data(inference_path):
    df = pd.read_csv(inference_path)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor

def run_inference(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs[0])
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
    return predictions

if __name__ == "__main__":
    input_dim = 4  
    hidden_dim = 16
    output_dim = 3  
    model = IrisClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    X_infer = load_inference_data(INFERENCE_PATH)
    infer_dataset = TensorDataset(X_infer)
    infer_loader = DataLoader(infer_dataset, batch_size=16, shuffle=False)
    
    predictions = run_inference(model, infer_loader)
    
    output_path = os.path.join(ROOT_DIR, '../data_process/data/inference_results.csv')
    pd.DataFrame(predictions, columns=['predictions']).to_csv(output_path, index=False)
    print(f'Inference results saved to {output_path}')