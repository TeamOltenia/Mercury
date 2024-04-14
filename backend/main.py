import joblib as joblib
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
import sklearn
import pickle
import numpy as np
import psycopg2
import asyncpg

import os
import json


MAILJET_API_KEY = "17f26b0d7fd61fe07a34cdeea7f1fbac"
MAILJET_API_SECRET = "14ea81b210dd1b904b9af8189474b2cd"
MAILJET_API_URL = "https://api.mailjet.com/v3.1/send"

app = FastAPI()
data_path = 'dataset/data.csv'
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",

]

dummy_data = pd.read_csv(data_path)

dummy_data = dummy_data[['merchant','amt','unix_time','is_fraud']]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_prediction = joblib.load('models/finalized_model.pkl')

class Transaction(BaseModel):
    c_num: int
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    street: str
    city: str
    state: str
    zip: int
    unix_time: int


# try:
#     model = joblib.load('models/finalized_model.pkl')
# except FileNotFoundError:
#     raise HTTPException(status_code=500, detail="Model file not found.")


with open('models/finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)




categorical_columns = ['merchant', 'category', 'first', 'last', 'street', 'city', 'state']


encoders = {col: joblib.load(f'models/{col}_encoder.pkl') for col in categorical_columns}

with open('models/model_zip-amt-time.pkl', 'rb') as file:
    loaded_model_zip_amt_time = pickle.load(file)
with open('models/scaler_zip-amt-time.pkl', 'rb') as file:
    loaded_scaler_zip_amt_time = pickle.load(file)

try:
    with open('models/model_card_number.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

@app.get("/dummy_data")
async def get_dummy_data(n: int = 100):
    dummy_data = pd.read_csv(data_path, nrows=n).to_dict(orient="records")
    return dummy_data


# @app.get("/data/{id}")
# async def get_data_by_id(id: str):  # Convert id to string
#     # Convert 'nameDest' column to string if necessary
#     dummy_data['nameDest'] = dummy_data['nameDest'].astype(str)
#
#     # Filter records
#     filtered_records = dummy_data[dummy_data['nameDest'] == id]
#
#     # Convert filtered records to dictionary
#     filtered_records_dict = filtered_records.to_dict(orient="records")
#
#     return filtered_records_dict




# Assuming you have the DATABASE_URL environment variable set:
async def connect_to_db():
    return await asyncpg.connect(
        "postgresql://admin:HMPKCWVd4i5t@ep-holy-night-a2ln8lgi.eu-central-1.aws.neon.tech/Mercury?sslmode=require"
    )


# async def get_data_db(merchant:str):
#      await connect_to_db.execute('''
#        SELECT merchant, amt, unix_time, is_fraud FROM log_after where merchant = $1
#     ''',merchant)
#

@app.get("/merchant/{merchant}")
async def get_merchant(merchant: str):
    try:
        # Connect to the database and execute the query
        connection = await connect_to_db()
        query = 'SELECT merchant, amt, unix_time, is_fraud FROM logs_after WHERE merchant = $1'
        rows = await connection.fetch(query, merchant)

        # Close the database connection
        await connection.close()

        # Convert rows to list of dicts
        data = [dict(row) for row in rows]

        # Check if data is empty and return a message or the data
        if not data:
            return {"message": "No data found for this merchant"}
        return data
    except Exception as e:
        # Handle exceptions by returning an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/{id}")
async def get_data_by_id(merchant: str):  # Convert id to string
    # Convert 'nameDest' column to string if necessary
    # dummy_data['nameDest'] = dummy_data['nameDest'].astype(str)

    # Filter records
    filtered_records = dummy_data[dummy_data['merchant'] == merchant]

        # Create a cursor object


    try:
        # Connect to the database and execute the query
        connection = await connect_to_db()
        query = 'SELECT merchant, amt, unix_time, is_fraud FROM logs_after WHERE merchant = $1'
        rows = await connection.fetch(query, merchant)

        # Close the database connection
        await connection.close()

        # Convert rows to list of dicts
        data = [dict(row) for row in rows]
        df = pd.DataFrame(data, columns=["merchant", "amt", "unix_time", "is_fraud"])

        filtered_records = pd.merge(filtered_records, df)

        # Check if data is empty and return a message or the data
        if not data:
            return {"message": "No data found for this merchant"}
        return data
    except Exception as e:
        # Handle exceptions by returning an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))

    # Create a DataFrame from the fetched data

    # Convert filtered records to dictionary
    filtered_records_dict = filtered_records.to_dict(orient="records")

    return filtered_records_dict

@app.get("/send_email")
async def send_email(to: str, subject: str, text: str):
    payload = {
        "Messages": [
            {
                "From": {"Email": "fazalunga404@gmail.com", "Name": "Faza Lunga"},
                "To": [{"Email": to, "Name": to}],
                "Subject": subject,
                "TextPart": text,
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            MAILJET_API_URL,
            auth=(MAILJET_API_KEY, MAILJET_API_SECRET),
            json=payload,
        )

        if response.status_code == 200:
            return JSONResponse(content={"message": "Email sent successfully"})
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to send email. Mailjet API response: {response.text}",
            )


def load_and_predict(zip_code, transaction_amount, unix_time):
    # Load the model and scaler


    # Prepare the new data using DataFrame to include feature names
    new_data = pd.DataFrame({
        'zip_code': [zip_code],
        'transaction_amount': [transaction_amount],
        'unix_time': [unix_time]
    })

    # Scale the new data using the loaded scaler
    new_data_scaled = loaded_scaler_zip_amt_time.transform(new_data)

    # Predict using the loaded model
    prediction = loaded_model_zip_amt_time.predict(new_data_scaled)

    # Return the predicted class and probabilities
    return 1 if prediction[0] == 1 else 0


def feature_engineering(card_number):
    """ Extracts features from the credit card number for the model. """
    return {
        'length': len(str(card_number)),
        'first_digit': int(str(card_number)[0]) if len(str(card_number)) > 0 else 0,
        'last_digit': int(str(card_number)[-1]) if len(str(card_number)) > 0 else 0,
        'digit_sum': sum(int(digit) for digit in str(card_number))
    }

def predict_fraud(card):
    # Extract features from the provided credit card number
    features = feature_engineering(card)
    feature_df = pd.DataFrame([features])
    # Predict using the trained model
    prediction = model.predict(feature_df)
    # Return the prediction result
    return  bool(prediction[0])

def luhn_check(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))
    return checksum % 10 == 0


@app.post("/predict/")
async def make_prediction(transaction: Transaction):
    try:
        # Initialize a dictionary to hold encoded features
        prob = load_and_predict(transaction.zip,transaction.amt,transaction.unix_time)
        if prob == 1:
            return {"prediction":   [1]}

        if not luhn_check(transaction.c_num):
            return {"prediction": [1]}

        encoded_features = {}
        for col in encoders.keys():
            attribute_value = getattr(transaction, col, None)  # Safely get attribute; use None if not found
            if col in encoders and attribute_value is not None:
                try:
                    # Encode the attribute if possible
                    encoded_features[col] = encoders[col].transform([attribute_value])[0]
                except Exception as e:
                    # Log and use a fallback value if the attribute cannot be encoded
                    print(f"Warning: Unable to encode {col} - {e}")
                    encoded_features[col] = np.random.randint(1,300)  # Default or fallback value
            else:
                # Use a default value if the encoder is missing or attribute is None
                encoded_features[col] = None


        # Assemble input data for the model, using `get` to handle any missing encoded features
        input_data = [
            encoded_features.get('merchant', None),
            encoded_features.get('category', None),
            encoded_features.get('first', None),
            encoded_features.get('last', None),
            encoded_features.get('street', None),
            encoded_features.get('city', None),
            encoded_features.get('state', None),
            transaction.zip,
            transaction.amt,
            transaction.c_num,
            transaction.unix_time
            # Assume other numerical fields are added here
        ]

        # Load the model (make sure to load it outside of request handling in a real application)

        prediction = model_prediction.predict([input_data])

        # Return the prediction as a list
        return {"prediction": prediction.tolist()}
    except Exception as e:
        # Raise an HTTPException with a 500 status code if any error occurs during processing
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
   # print(predict_fraud(30569309025904))
