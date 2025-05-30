"""
train.py

This script trains the machine learning model using the dataset and saves the trained model to the models directory.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
data = pd.read_csv('../input/your_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/random_forest_model.pkl')
print("Model trained and saved successfully.")
