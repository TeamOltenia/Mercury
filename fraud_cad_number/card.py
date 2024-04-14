import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Function to check if the card number passes the Luhn algorithm
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

# Feature engineering function
def feature_engineering(df):
    # Ensure CardNumber is a string and handle NaN values if any
    df['CardNumber'] = df['CardNumber'].astype(str).replace('nan', np.nan)

    # Drop rows where CardNumber is NaN
    df = df.dropna(subset=['CardNumber'])

    df['card_length'] = df['CardNumber'].apply(len)
    df['luhn_valid'] = df['CardNumber'].apply(luhn_check)
    df['first_digit'] = df['CardNumber'].apply(lambda x: int(x[0]))
    df['luhn_valid'] = df['luhn_valid'].astype(int)  # Convert boolean to int
    return df

# Load your data (this path needs to be updated to your actual file location)
data = pd.read_csv('card.csv')

# Assuming you have a column named 'IssuingNetwork' and 'CardNumber'
# And assuming 'valid' is a column that you have or can generate
data = feature_engineering(data)

# Prepare features and labels
X = data[['card_length', 'luhn_valid', 'first_digit']]
y = data['valid']  # Assuming 'valid' is a column indicating if the card is valid (0 or 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
