import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle

np.random.seed(0)

# Define dataset size and proportions
data_size = 10000
fraudulent_prop = 0.1
clean_prop = 1 - fraudulent_prop
fraudulent_size = int(data_size * fraudulent_prop)
clean_size = data_size - fraudulent_size

# Helper function to generate random zip codes
def generate_zip_codes(n):
    # Generate zip codes ranging from 10000 to 99999
    return [str(np.random.randint(10000, 100000)) for _ in range(n)]

# Generate clean transactions
clean_data = pd.DataFrame({
    'zip_code': generate_zip_codes(clean_size),
    'transaction_amount': np.random.exponential(scale=150, size=clean_size),
    'unix_time': np.array([int((datetime.now() - timedelta(days=np.random.randint(1, 365))).timestamp()) for _ in range(clean_size)]),
    'is_fraud': np.zeros(clean_size, dtype=int)
})

# Generate fraudulent transactions
fraudulent_data = pd.DataFrame({
    'zip_code': generate_zip_codes(fraudulent_size),
    'transaction_amount': np.concatenate([np.random.exponential(scale=500, size=int(fraudulent_size * 0.5)),
                                          np.full(int(fraudulent_size * 0.5), np.random.randint(1000, 5000))]),
    'unix_time': np.concatenate([np.array([int((datetime.now() - timedelta(minutes=np.random.randint(1, 60))).timestamp()) for _ in range(int(fraudulent_size * 0.5))]),
                                 np.array([int((datetime.now() - timedelta(days=np.random.randint(1, 10))).timestamp()) for _ in range(int(fraudulent_size * 0.5))])]),
    'is_fraud': np.ones(fraudulent_size, dtype=int)
})

# Combine and shuffle the dataset
data = pd.concat([clean_data, fraudulent_data], ignore_index=True).sample(frac=1).reset_index(drop=True)


data.to_csv("fraud_zip.csv")


print(data[data['is_fraud'] ==1].size)
print(data[data['is_fraud'] ==1].head())
print(data.head())
print(data.size)
