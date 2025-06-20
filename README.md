# 🔄 Customer Churn Prediction using Artificial Neural Network (ANN)

This project implements a deep learning model to predict whether a customer will **exit (churn)** from a bank based on structured customer profile data. The model uses an **Artificial Neural Network (ANN)** built with TensorFlow/Keras and applies comprehensive preprocessing to achieve high prediction accuracy.

---

## 📌 Key Features

- Developed a binary classification model using **Keras Sequential API**.
- Performed **label encoding**, **one-hot encoding**, and **feature scaling** for preprocessing.
- Built a **fully connected feedforward neural network (ANN)**.
- Trained and evaluated the model using real-world bank customer data.
- Achieved high accuracy in predicting customer churn behavior.

---

## 📂 Dataset Overview

**Dataset Name**: `Churn_Modelling.csv`  
**Source**: [Kaggle – Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shubhendra7/customer-churn-prediction)

**Features:**
- **Customer Information**: CreditScore, Age, Tenure, Balance, etc.
- **Categorical**: Geography, Gender
- **Target**: `Exited` (1 → Churned, 0 → Retained)

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Extracted input features (X) and target (y).
- Applied **Label Encoding** on `Gender` and **One-Hot Encoding** on `Geography`.
- Used **StandardScaler** for feature normalization.

### 2. Model Building
- Built a **3-layer neural network**:
  - Two hidden layers with ReLU activation.
  - One output layer with Sigmoid activation for binary classification.

### 3. Training
- Compiled the model using:
  - Optimizer: `adam`
  - Loss: `binary_crossentropy`
- Trained over 100 epochs with batch size = 10.

### 4. Evaluation
- Generated predictions and evaluated with:
  - **Confusion Matrix**
  - **Accuracy Score**

---

## 🧠 Model Architecture

| Layer Type      | Units | Activation | Description                        |
|-----------------|-------|------------|------------------------------------|
| Input + Dense 1 |   6   | ReLU       | First hidden layer (input_dim=11)  |
| Dense 2         |   6   | ReLU       | Second hidden layer                |
| Output          |   1   | Sigmoid    | Output layer for binary outcome    |

---

## 🧾 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
