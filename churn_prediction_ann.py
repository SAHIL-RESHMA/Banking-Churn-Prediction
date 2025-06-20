"""
Customer Churn Prediction using Artificial Neural Network (ANN)

This script builds a binary classification model using a deep learning approach (ANN)
to predict whether a customer will churn (exit) from a bank based on structured profile data.
"""

# ==============================================================================
# 📦 Import Required Libraries
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==============================================================================
# 📥 Load Dataset
# ==============================================================================
DATA_PATH = "data/Churn_Modelling.csv"
df = pd.read_csv(DATA_PATH)

# ==============================================================================
# 🎯 Feature and Target Selection
# ==============================================================================
# Independent features: columns 3 to 12
X = df.iloc[:, 3:13].values

# Dependent target variable: Exited column
y = df.iloc[:, 13].values

# ==============================================================================
# 🔄 Encode Categorical Variables
# ==============================================================================
# Label encode Gender (Male/Female → 1/0)
gender_encoder = LabelEncoder()
X[:, 2] = gender_encoder.fit_transform(X[:, 2])

# One-hot encode Geography (France, Spain, Germany)
geo_encoder = ColumnTransformer(
    transformers=[('geo', OneHotEncoder(), [1])],
    remainder='passthrough'
)
X = geo_encoder.fit_transform(X)
X = X[:, 1:].astype(float)  # Avoid dummy variable trap

# ==============================================================================
# ✂️ Train-Test Split
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================================================================
# ⚖️ Feature Scaling
# ==============================================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================================================================
# 🧠 Build ANN Model
# ==============================================================================
model = Sequential()
model.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=X_train.shape[1]))
model.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==============================================================================
# 🚀 Train the Model
# ==============================================================================
history = model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1)

# ==============================================================================
# 📈 Predictions and Evaluation
# ==============================================================================
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Metrics
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\n📊 Confusion Matrix:\n", cm)
print(f"✅ Accuracy: {acc:.4f}")
