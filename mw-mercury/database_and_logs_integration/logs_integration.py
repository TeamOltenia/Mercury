import json
import asyncpg
import asyncio
from pydantic import BaseModel

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

async def connect_to_db(retry_attempts=5, retry_delay=2):
    """Attempt to connect to the database with retries."""
    last_exception = None
    for attempt in range(retry_attempts):
        try:
            # Attempt to establish a database connection
            return await asyncpg.connect(
                "postgresql://admin:HMPKCWVd4i5t@ep-holy-night-a2ln8lgi.eu-central-1.aws.neon.tech/Mercury?sslmode=require"
            )
        except Exception as e:
            last_exception = e
            print(f"Failed to connect to the database, attempt {attempt + 1}/{retry_attempts}. Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
    
    # If all attempts fail, raise the last caught exception
    raise Exception(f"All connection attempts failed: {last_exception}")

async def insert_log_before(conn, log: LogBefore):
    await conn.execute('''
        INSERT INTO Logs_Before (cc_num, merchant, category, amt, first, last, street, city, state, zip, unix_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    ''', log.cc_num, log.merchant, log.category, log.amt, log.first, log.last,
    log.street, log.city, log.state, log.zip, log.unix_time)

async def insert_log_after(conn, log: LogAfter):
    await conn.execute('''
        INSERT INTO Logs_After (cc_num, merchant, category, amt, first, last, street, city, state, zip, unix_time, is_fraud)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ''', log.cc_num, log.merchant, log.category, log.amt, log.first, log.last,
    log.street, log.city, log.state, log.zip, log.unix_time, log.is_fraud)

async def add_log(table_name, request_body):
    conn = await connect_to_db()
    try:
        if table_name == "Logs_Before":
            log_before = LogBefore(**request_body)
            await insert_log_before(conn, log_before)
        elif table_name == "Logs_After":
            log_after = LogAfter(**request_body)
            await insert_log_after(conn, log_after)
        else:
            raise ValueError("Invalid table name. It must be 'Logs_Before' or 'Logs_After'.")
        return {"message": f"Log added to {table_name} table successfully"}
    finally:
        await conn.close()

def lambda_handler(event, context):
    loop = asyncio.get_event_loop()
    try:
        event = json.loads(event.get('body', '{}'))
        table_name = event['table_name']
        request_body = event['request_body']
        response = loop.run_until_complete(add_log(table_name, request_body))
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }


# Mock data for testing
logs_before_mock = {
    "cc_num": 1234567890123456,
    "merchant": "Amazon",
    "category": "Retail",
    "amt": 100.50,
    "first": "John",
    "last": "Doe",
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "zip": 10001,
    "unix_time": 1649862000
}

logs_after_mock = {
    "cc_num": 9876543210987654,
    "merchant": "eBay",
    "category": "Online Marketplace",
    "amt": 500.75,
    "first": "Jane",
    "last": "Smith",
    "street": "456 Elm St",
    "city": "Los Angeles",
    "state": "CA",
    "zip": 90001,
    "unix_time": 1649865600,
    "is_fraud": True
}

# async def test():
#     # Testing inserting log before
#     print("Testing inserting log before...")
#     await lambda_handler({'table_name': 'Logs_Before', 'request_body': json.dumps(logs_before_mock)}, None)

#     # Testing inserting log after
#     print("Testing inserting log after...")
#     await lambda_handler({'table_name': 'Logs_After', 'request_body': json.dumps(logs_after_mock)}, None)
