import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from data_process.data_processing import IrisSetGenerator, get_project_dir
from training.train import load_data, IrisClassifier, train_model
from inference.run import load_inference_data, run_inference

# Testing the data processing first
class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.generator = IrisSetGenerator()
        self.test_data_path = 'test_train_data.csv'

    def tearDown(self):
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)

    # Test for getting project directory
    def test_get_project_dir(self):
        result = get_project_dir('data')
        self.assertTrue(os.path.isdir(result))

    # Test creating labeled data
    def test_create_labeled_data(self):
        df = self.generator.create(is_labeled=True, save_path=self.test_data_path)
        self.assertFalse(df.empty)
        self.assertIn('target', df.columns)

    # Test creating unlabeled data
    def test_create_unlabeled_data(self):
        df = self.generator.create(is_labeled=False)
        self.assertFalse(df.empty)
        self.assertNotIn('target', df.columns)

# Testing the training process
class TestTraining(unittest.TestCase):

    def setUp(self):
        self.data_path = 'test_train_data.csv'
        self.model_save_path = 'test_model.pth'
        self.df = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9, 4.7, 4.6, 5.0],
            'sepal width (cm)': [3.5, 3.0, 3.2, 3.1, 3.6],
            'petal length (cm)': [1.4, 1.4, 1.3, 1.5, 1.4],
            'petal width (cm)': [0.2, 0.2, 0.2, 0.2, 0.2],
            'target': [0, 1, 1, 0, 2]
        })
        self.df.to_csv(self.data_path, index=False)

    def tearDown(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        if os.path.exists(self.model_save_path):
            os.remove(self.model_save_path)

    # Test loading training data
    def test_load_data(self):
        X_train, X_test, y_train, y_test = load_data(self.data_path)
        self.assertEqual(X_train.shape[1], 4)
        self.assertEqual(len(y_train), 4)

    # Test training model and saving it
    def test_model_training(self):
        X_train, _, y_train, _ = load_data(self.data_path)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        model = IrisClassifier(4, 16, 3)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_model(model, criterion, optimizer, train_loader, num_epochs=2)
        torch.save(model.state_dict(), self.model_save_path)

        self.assertTrue(os.path.exists(self.model_save_path))
        
# Testing the inference process
class TestInference(unittest.TestCase):

    def setUp(self):
        self.data_path = 'test_inference_data.csv'
        self.model_path = 'test_model.pth'
        self.output_path = 'test_inference_results.csv'

        self.df = pd.DataFrame({
            'sepal length (cm)': [5.1, 4.9],
            'sepal width (cm)': [3.5, 3.0],
            'petal length (cm)': [1.4, 1.4],
            'petal width (cm)': [0.2, 0.2]
        })
        self.df.to_csv(self.data_path, index=False)

        model = IrisClassifier(4, 16, 3)
        torch.save(model.state_dict(), self.model_path)

    def tearDown(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    # Test loading inference data
    def test_load_inference_data(self):
        X_tensor = load_inference_data(self.data_path)
        self.assertEqual(X_tensor.shape[1], 4)

    # Test running inference on the model
    def test_run_inference(self):
        model = IrisClassifier(4, 16, 3)
        model.load_state_dict(torch.load(self.model_path))
        X_infer = load_inference_data(self.data_path)
        infer_dataset = TensorDataset(X_infer)
        infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False)

        predictions = run_inference(model, infer_loader)
        self.assertEqual(len(predictions), len(self.df))

if __name__ == '__main__':
    unittest.main()
