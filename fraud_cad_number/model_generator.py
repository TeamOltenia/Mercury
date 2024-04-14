# import pandas as pd
# import numpy as np
# import random
#
# # Constants
# n = 10000  # Total number of records
# fraud_ratio = 0.2  # 20% fraud
#
# # Helper function to generate a random credit card number
# def generate_credit_card():
#     return ''.join([str(random.randint(0, 9)) for _ in range(16)])
#
# # Helper function to generate an invalid credit card number for fraud cases
# def generate_invalid_credit_card():
#     type_of_error = random.choice(['length', 'pattern'])
#     if type_of_error == 'length':
#         # Generate a card number with length not equal to 16
#         length = random.choice([14, 15, 17, 18])
#         return ''.join([str(random.randint(0, 9)) for _ in range(length)])
#     else:
#         # Generate a patterned card where some digits are the same
#         repeated_sequence = ''.join([str(random.randint(0, 9)) for _ in range(4)])
#         # Place the repeated sequence at the start, end, or in an 8-digit block
#         position = random.choice(['start', 'end', 'block'])
#         if position == 'start':
#             return repeated_sequence * 4
#         elif position == 'end':
#             return generate_credit_card()[:8] + repeated_sequence * 2
#         else:
#             return generate_credit_card()[:4] + repeated_sequence * 2 + generate_credit_card()[:4]
#
# # Generating data
# data = {
#     'credit_card_number': [],
#     'is_fraud': []
# }
#
# # Assign fraud labels
# is_fraud = np.array([1] * int(n * fraud_ratio) + [0] * int(n * (1 - fraud_ratio)))
# np.random.shuffle(is_fraud)
#
# # Generate credit card numbers based on fraud label
# for fraud in is_fraud:
#     if fraud == 1:
#         data['credit_card_number'].append(generate_invalid_credit_card())
#     else:
#         data['credit_card_number'].append(generate_credit_card())
#     data['is_fraud'].append(fraud)
#
# # Create DataFrame
# df = pd.DataFrame(data)
#
# df.to_csv('card_fraud.csv')
#
# # Preview the DataFrame
# print(df.head())
# print("\nFraud Distribution:\n", df['is_fraud'].value_counts())
#
# # Optionally, save to CSV
# # df.to_csv('credit_card_data.csv', index=False)


import pandas as pd
import numpy as np
import random
import pickle

# Constants for dataset
n = 500000  # Total number of records
fraud_ratio = 0.2  # 20% fraud

def generate_credit_card():
    """ Generates a valid 16-digit credit card number. """
    return ''.join([str(random.randint(0, 9)) for _ in range(16)])

def generate_invalid_credit_card():
    """ Generates an invalid credit card number for fraud cases. """
    type_of_error = random.choice(['length', 'pattern'])
    if type_of_error == 'length':
        length = random.choice([14, 15, 17, 18])
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])
    else:
        repeated_sequence = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        position = random.choice(['start', 'end', 'block'])
        if position == 'start':
            return repeated_sequence * 4
        elif position == 'end':
            return generate_credit_card()[:8] + repeated_sequence * 2
        else:
            return generate_credit_card()[:4] + repeated_sequence * 2 + generate_credit_card()[:4]

# Generate dataset
data = {'id': range(1, n + 1), 'credit_card_number': [], 'is_fraud': []}
is_fraud = np.array([1] * int(n * fraud_ratio) + [0] * int(n * (1 - fraud_ratio)))
np.random.shuffle(is_fraud)

for fraud in is_fraud:
    if fraud == 1:
        data['credit_card_number'].append(generate_invalid_credit_card())
    else:
        data['credit_card_number'].append(generate_credit_card())
    data['is_fraud'].append(fraud)

df = pd.DataFrame(data)


def feature_engineering(df):
    """ Extracts features from the credit card number. """
    df['length'] = df['credit_card_number'].apply(len)
    df['first_digit'] = df['credit_card_number'].apply(lambda x: int(x[0]) if len(x) > 0 else 0)
    df['last_digit'] = df['credit_card_number'].apply(lambda x: int(x[-1]) if len(x) > 0 else 0)
    df['digit_sum'] = df['credit_card_number'].apply(lambda x: sum(int(digit) for digit in x))
    return df

df = feature_engineering(df)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Splitting the dataset
X = df[['length', 'first_digit', 'last_digit', 'digit_sum']]
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Optionally check accuracy on test set
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


filename = 'finalized_model2.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
