from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg

app = FastAPI()

class LogBefore(BaseModel):
    cc_num: int
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

class LogAfter(BaseModel):
    cc_num: int
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
    is_fraud: bool

async def connect_to_db():
    return await asyncpg.connect(
        "postgresql://admin:HMPKCWVd4i5t@ep-holy-night-a2ln8lgi.eu-central-1.aws.neon.tech/Mercury?sslmode=require"
    )

async def create_tables(conn):
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS Logs_Before (
            id SERIAL PRIMARY KEY,
            cc_num BIGINT,
            merchant VARCHAR(255),
            category VARCHAR(255),
            amt FLOAT,
            first VARCHAR(255),
            last VARCHAR(255),
            street VARCHAR(255),
            city VARCHAR(255),
            state VARCHAR(255),
            zip BIGINT,
            unix_time BIGINT
        )
    ''')

    await conn.execute('''
        CREATE TABLE IF NOT EXISTS Logs_After (
            id SERIAL PRIMARY KEY,
            cc_num BIGINT,
            merchant VARCHAR(255),
            category VARCHAR(255),
            amt FLOAT,
            first VARCHAR(255),
            last VARCHAR(255),
            street VARCHAR(255),
            city VARCHAR(255),
            state VARCHAR(255),
            zip BIGINT,
            unix_time BIGINT,
            is_fraud BOOLEAN
        )
    ''')

async def insert_log_before(conn, log: LogBefore):
    await conn.execute('''
        INSERT INTO Logs_Before (
            cc_num, merchant, category, amt, first, last, street, city, state, zip, unix_time
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    ''', log.cc_num, log.merchant, log.category, log.amt, log.first, log.last,
    log.street, log.city, log.state, log.zip, log.unix_time)

async def insert_log_after(conn, log: LogAfter):
    await conn.execute('''
        INSERT INTO Logs_After (
            cc_num, merchant, category, amt, first, last, street, city, state, zip, unix_time, is_fraud
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ''', log.cc_num, log.merchant, log.category, log.amt, log.first, log.last,
    log.street, log.city, log.state, log.zip, log.unix_time, log.is_fraud)

@app.post("/logs/before/")
async def add_log_before(log: LogBefore):
    conn = await connect_to_db()
    try:
        await create_tables(conn)
        # await insert_log_before(conn, log)
        return {"message": "Log added to Logs_Before table successfully"}
    finally:
        await conn.close()

@app.post("/logs/after/")
async def add_log_after(log: LogAfter):
    conn = await connect_to_db()
    try:
        await create_tables(conn)
        # await insert_log_after(conn, log)
        return {"message": "Log added to Logs_After table successfully"}
    finally:
        await conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)