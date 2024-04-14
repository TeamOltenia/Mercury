from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

# Define the request body model
class CreditCard(BaseModel):
    number: str

# Load the model from a pickle file
try:
    with open('finalized_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

def feature_engineering(card_number):
    """ Extracts features from the credit card number for the model. """
    return {
        'length': len(card_number),
        'first_digit': int(card_number[0]) if len(card_number) > 0 else 0,
        'last_digit': int(card_number[-1]) if len(card_number) > 0 else 0,
        'digit_sum': sum(int(digit) for digit in card_number)
    }

@app.post("/predict_fraud/")
async def predict_fraud(card: CreditCard):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    try:
        # Extract features from the provided credit card number
        features = feature_engineering(card.number)
        feature_df = pd.DataFrame([features])
        # Predict using the trained model
        prediction = model.predict(feature_df)
        # Return the prediction result
        return {"is_fraudulent": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
