"""
utils.py

This module contains utility functions for data preprocessing and evaluation.
"""

import pandas as pd
from sklearn.metrics import accuracy_score

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def evaluate_model(model, X, y):
    """Evaluate the model and print accuracy."""
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
