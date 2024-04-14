from datetime import datetime
import pickle
import numpy as np
import pandas as pd

def load_and_predict(zip_code, transaction_amount, unix_time):
    # Load the model and scaler
    with open('model_zip-amt-time.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('scaler_zip-amt-time.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)

    # Prepare the new data using DataFrame to include feature names
    new_data = pd.DataFrame({
        'zip_code': [zip_code],
        'transaction_amount': [transaction_amount],
        'unix_time': [unix_time]
    })

    # Scale the new data using the loaded scaler
    new_data_scaled = loaded_scaler.transform(new_data)

    # Predict using the loaded model
    prediction = loaded_model.predict(new_data_scaled)

    # Optionally, return probability estimates for classes
    probabilities = loaded_model.predict_proba(new_data_scaled)

    # Return the predicted class and probabilities
    return ("Fraudulent" if prediction[0] == 1 else "Clean", probabilities)

# Example usage of the function
example_zip_code = 231414  # Example integer zip code
example_amount = 340  # Example transaction amount
example_time = int(datetime.now().timestamp())  # Example Unix time

# Getting prediction and probability
prediction, probabilities = load_and_predict(example_zip_code, example_amount, example_time)
print("The transaction is predicted to be:", prediction)
print("Probability [Clean, Fraudulent]:", probabilities)
