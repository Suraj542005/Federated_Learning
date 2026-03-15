
# Federated Learning with CNN for MNIST Digit Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Flower](https://img.shields.io/badge/FederatedLearning-Flower-green)

---

# Overview

This project demonstrates a **Privacy-Preserving Federated Learning system** using **Flower** and **TensorFlow**.

Instead of collecting data from multiple users in one central location, **each client trains a model locally and sends only the model updates to a central server**. The server aggregates the updates to create a **global model**.

The example problem used in this repository is **handwritten digit classification using the MNIST dataset**.

The system simulates **two independent clients** training a shared model collaboratively without sharing their raw data.

---

# What is Federated Learning?

Traditional Machine Learning works like this:

User Data → Central Server → Model Training

This approach raises **privacy and security concerns** because all data must be uploaded to the server.

Federated Learning changes this process:

Client Data → Local Training → Model Updates → Server Aggregation → Global Model

Key idea:

- Data never leaves the client device
- Only model parameters (weights) are shared

---

# System Architecture

```
                +----------------------+
                |        SERVER        |
                |  Global CNN Model    |
                | Federated Averaging  |
                +----------+-----------+
                           |
                    Model Aggregation
                           |
        +------------------+------------------+
        |                                     |
   +-----------+                         +-----------+
   | Client 1  |                         | Client 2  |
   | Local CNN |                         | Local CNN |
   | MNIST A   |                         | MNIST B   |
   +-----------+                         +-----------+
```

---

# Project Structure

```
FL_version1
│
├── clients
│   ├── client1.py
│   └── client2.py
│
├── server
│   ├── server.py
│   └── strategy.py
│
├── models
│   └── cnn_model.py
│
├── data
│   └── mnist_loader.py
│
├── inference
│   ├── predict_mnist.py
│   └── custom_predict.py
│
├── custom_images
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# CNN Model Architecture

The model used for training is a **Convolutional Neural Network (CNN)** designed for digit recognition.

Architecture:

Input Layer (28x28x1)  
         ↓  
  Conv2D (32 filters)  
         ↓  
     MaxPooling  
         ↓  
  Conv2D (64 filters)  
         ↓  
     MaxPooling  
         ↓  
      Flatten  
         ↓  
  Dense (128 neurons)  
         ↓   
Output Layer (10 classes)

---

# Dataset

The dataset used in this project is **MNIST**.

Dataset size:

Training images: 60,000  
Testing images: 10,000  

Each image is:

28 × 28 grayscale image

Dataset is automatically downloaded using TensorFlow.

---

# Federated Learning Workflow

Step 1: Server initializes global model.  
Step 2: Global model is sent to all clients.  
Step 3: Clients train locally on their private datasets.  
Step 4: Clients send model updates to the server.  
Step 5: Server performs Federated Averaging.  
Step 6: Updated global model is redistributed.  
Step 7: Process repeats for multiple rounds.

---

# Installation Guide

## Clone the Repository

git clone https://github.com/Suraj542005/Federated_Learning.git

cd Federated_Learning

## Install Dependencies

pip install -r requirements.txt

---

# Running the Federated Learning System

Open three terminals.

### Start Server

python -m server.server

### Start Client 1

python -m clients.client1

### Start Client 2

python -m clients.client2

---

# Running Predictions

### MNIST Prediction

python inference/predict.py

---

# Privacy Advantages

- Data remains on client devices
- Only model updates are shared
- Improved privacy and security

---

# Future Research Directions

- Differential Privacy
- Secure Aggregation
- Federated Learning Dashboard
- Medical dataset training
- Edge device deployment

---

# Technologies Used

Python  
TensorFlow  
Flower (Federated Learning Framework)  
NumPy  
Matplotlib  
Pillow  

---

# Developer

Suraj Jagtap  
> "Train models together, keep data private."
