"""
predict.py

This script loads the trained model and makes predictions on new data.
"""

import pandas as pd
import joblib

# Load the trained model
model = joblib.load('../models/random_forest_model.pkl')

# Load new data
new_data = pd.read_csv('../input/new_data.csv')

# Make predictions
predictions = model.predict(new_data)

# Save predictions
output = pd.DataFrame({'Id': new_data.index, 'Prediction': predictions})
output.to_csv('../input/predictions.csv', index=False)
print("Predictions saved successfully.")
