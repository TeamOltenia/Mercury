import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
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

@app.get("/data/{id}")
async def get_data_by_id(merchant: str):  # Convert id to string
    # Convert 'nameDest' column to string if necessary
    # dummy_data['nameDest'] = dummy_data['nameDest'].astype(str)

    # Filter records
    filtered_records = dummy_data[dummy_data['merchant'] == merchant]

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
   # print(predict_fraud(30569309025904))
