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
data_path = '../models/onlinefraud.csv'
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/dummy_data")
async def get_dummy_data(n: int = 100):
    dummy_data = pd.read_csv(data_path, nrows=n).to_dict(orient="records")
    return dummy_data

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
