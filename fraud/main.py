from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib


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


app = FastAPI()

# Load your trained model
try:
    model = joblib.load('finalized_model.pkl')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found.")

categorical_columns = ['merchant', 'category', 'first', 'last', 'street', 'city', 'state']


encoders = {col: joblib.load(f'{col}_encoder.pkl') for col in categorical_columns}


@app.post("/predict/")
async def make_prediction(transaction: Transaction):
    try:
        # Encode all categorical features
        encoded_features = {
            col: encoders[col].transform([getattr(transaction, col)])[0]
            for col in encoders.keys()
        }

        # Add other transaction fields, and assemble input for model
        input_data = [
            encoded_features['merchant'],
            encoded_features['category'],
            encoded_features['first'],
            encoded_features['last'],
            encoded_features['street'],
            encoded_features['city'],
            encoded_features['state'],
            transaction.zip,
            transaction.amt,
            transaction.c_num,
            transaction.unix_time
            # Assume other numerical fields are added here
        ]

        # Load the model (make sure to load it outside of request handling in a real application)
        model = joblib.load('finalized_model.pkl')
        prediction = model.predict([input_data])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)