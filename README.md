# 🔄 Customer Churn Prediction using Artificial Neural Network (ANN)

This project implements a deep learning model to predict whether a customer will **exit (churn)** from a bank based on structured customer profile data. The model uses an **Artificial Neural Network (ANN)** built with TensorFlow/Keras and applies comprehensive preprocessing to achieve reliable prediction accuracy.

---

## 📌 Key Features

- Developed a binary classification model using the **Keras Sequential API**.
- Applied proper **data encoding** and **feature scaling** techniques.
- Built a **fully connected feedforward neural network** (ANN).
- Trained and validated using a real-world banking dataset.
- Evaluated model performance using a **confusion matrix** and **accuracy score**.

---

## 📂 Dataset Overview

- **Dataset**: `Churn_Modelling.csv`

### ✨ Key Columns:
- `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary` – Numerical
- `Geography`, `Gender` – Categorical
- `Exited` – Target variable (1 = Churned, 0 = Retained)

---

## ⚙️ Project Workflow

### 1. 📊 Data Preprocessing
- Extracted input features (X) and target labels (y).
- **Label Encoding** for binary categorical feature (`Gender`).
- **One-Hot Encoding** for multi-class feature (`Geography`).
- **StandardScaler** used to normalize numerical values.

### 2. 🧠 Model Architecture
Built using `Sequential()` model:

| Layer Type      | Units | Activation | Description                        |
|-----------------|-------|------------|------------------------------------|
| Input + Dense 1 |   6   | ReLU       | First hidden layer (input_dim=11)  |
| Dense 2         |   6   | ReLU       | Second hidden layer                |
| Output Layer    |   1   | Sigmoid    | Binary classification output       |

### 3. 🏋️ Model Training
- **Optimizer**: `adam`
- **Loss Function**: `binary_crossentropy`
- **Epochs**: 100  
- **Batch Size**: 10  

### 4. 📈 Model Evaluation
- Predictions made on test data.
- Evaluation using **confusion matrix** and **accuracy score**.

---

## 📈 Sample Output

✅ Results shown below were generated after training the model for 100 epochs on CPU, demonstrating stable convergence and reasonable performance for binary classification.

📊 Confusion Matrix:
[[1595    0]
 [ 405    0]]

✅ Final Accuracy: 0.8036
