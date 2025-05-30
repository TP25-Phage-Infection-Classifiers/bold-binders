"""
tune_model.py

This script performs hyperparameter tuning on the selected model.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv('../input/your_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize model
model = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X, y)

print(f"Best parameters found: {grid_search.best_params_}")
