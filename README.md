# Iris Classification with Neural Network
Welcome to the Iris Classification project! This project demonstrates how to apply a simple neural network model using PyTorch to classify the Iris dataset, a classic dataset used in machine learning. The project is structured to handle everything from data processing to model training, and inference. It also includes unittests to ensure the reliability of each part of the pipeline.

## Prerequisites

Before diving into the detailed steps of setting up and using this project, there are few important prerequisites or requirements that need to be addressed. These prerequisites ensure that your local development environment is ready and capable of efficiently running and supporting the project. 

### Forking and Cloning from GitHub
Create a copy of this repository by forking it on GitHub.

Clone the forked repository to your local machine:

```bash
git clone https://github.com/yourusername/Iris-Classification.git
```

### Setting Up Development Environment
Ensure you have Python 3.8+ installed on your machine. 

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Also make sure to have Docker Desktop installed

## Project structure:

This project has a modular structure, where each folder has a specific duty.

```bash
Iris-Classification
├── data_process                 # Data processing and data handling
│   ├── data_processing.py       # Data processing scripts
│   ├── data                    # Contains train and inference data
│   └── __init__.py
├── inference                    # Inference pipeline
│   ├── Dockerfile               # Dockerfile for inference setup
│   ├── run.py                   # Run the inference
│   └── __init__.py
├── models                       # Folder to store trained models
├── training                     # Model training pipeline
│   ├── Dockerfile               # Dockerfile for training setup
│   ├── train.py                 # Training script
│   └── __init__.py
├── unittests                    # Unit tests for data processing, training, and inference
│   ├── unittests.py
│   └── __init__.py
├── .gitignore                   # Ignore unnecessary files for git
├── README.md                    # Project documentation
├── requirements.txt             # Project dependencies
├── settings.json                # Configuration file for settings
└── __init__.py

```

## Data Overview
This project uses the well-known Iris dataset from scikit-learn. The dataset contains 150 samples of iris flowers, categorized into three species: Setosa, Versicolor, and Virginica. Each sample has four features: sepal length, sepal width, petal length, and petal width.

## Data Processing
The data processing is handled by the `data_processing.py` script located in the `data_process` folder. It reads the Iris dataset, cleans it if necessary, and prepares it for training. It also splits the dataset into training and testing sets. The processed data is saved in the `data` directory, which will be used for training and inference.


## Model Training
The training of the neural network model is handled by the `train.py` script located in the `training` folder. A simple neural network built using PyTorch is trained on the Iris dataset. The model is saved in the `models` folder after training.

To train the model, simply run the following command:

```bash
python training/train.py
```

### Running Training with Docker
Build the Docker image for training:

```bash
docker build -f ./training/Dockerfile -t iris_training_image .
```

Run the Docker container to train the model:

```bash
docker run -it iris_training_image /bin/bash
```

## Inference 
Once the model is trained, it can be used for inference. The inference pipeline is implemented in the `run.py` script located in the `inference` folder.

To run the inference locally:

```bash
docker run -it iris_training_image /bin/bash
```

### Running Inference with Docker
Build the Docker image for inference:

```bash
docker build -f ./inference/Dockerfile --build-arg model_name=model.pth -t iris_inference_image .
```

Run the inference Docker container:

```bash
docker run -v /path_to_your_local_model:/app/models -v /path_to_input_data:/app/input -v /path_to_output_data:/app/output iris_inference_image
```

## Unit Testing
This project includes unit tests located in the `unittests/unittests.py` file. These tests ensure that the data processing, training, and inference components work correctly.
To run the tests:

```bash
python -m unittest unittests/unittests.py
```

## Settings
The `settings.json` file contains important configuration details such as file paths and hyperparameters used in the project. Make sure this file is correctly set up before running the project.

## Requirements
The project has the following Python dependencies, which are listed in the `requirements.txt` file:

- `torch==2.5.1`
- `numpy==2.2.2`
- `pandas==2.2.3`
- `scikit-learn==1.6.1`
- `scipy==1.15.1`

and other dependencies for smooth execution.

To install them, simply run:
```bash
pip install -r requirements.txt
```

## Wrap Up
This project demonstrates a straightforward application of neural networks on the Iris classification dataset. By following this setup, you should be able to process data, train models, run inference, and test each part of the pipeline with ease. Enjoy experimenting with the project and feel free to modify the neural network architecture for better results!