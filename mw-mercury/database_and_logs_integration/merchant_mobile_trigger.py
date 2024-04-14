import asyncio
import asyncpg
from pydantic import BaseModel

# Assuming previous classes and connection function

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

async def get_by_merchant(merchant: str):
    conn = await connect_to_db()
    try:
        # SQL query selecting specific fields for a given merchant
        records = await conn.fetch('''
            SELECT amt, unix_time, is_fraud FROM Logs_After
            WHERE merchant = $1
        ''', merchant)

        # Convert records to list of dictionaries
        result = [dict(record) for record in records]
        return json.dumps(result)  # Return as JSON string
    finally:
        await conn.close()

# async def test_get_by_merchant():
#     # Example usage of the get_by_merchant function
#     merchant_name = "eBay"
#     result = await get_by_merchant(merchant_name)
#     print(f"Logs for merchant '{merchant_name}': {result}")
#
# # Run the test function to see results
# asyncio.run(test_get_by_merchant())
