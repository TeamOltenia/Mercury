import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
# Generate a random dataset
def generate_dataset(data_size):
    fraudulent_prop = 0.2
    clean_size = int(data_size * (1 - fraudulent_prop))
    fraudulent_size = data_size - clean_size

    # Helper function to generate random zip codes as integers
    def generate_zip_codes(n):
        return np.random.randint(10000, 100000, size=n)

    clean_data = pd.DataFrame({
        'zip_code': generate_zip_codes(clean_size),
        'transaction_amount': np.random.exponential(scale=150, size=clean_size),
        'unix_time': np.array([int((datetime.now() - timedelta(days=np.random.randint(1, 365))).timestamp()) for _ in range(clean_size)]),
        'is_fraud': np.zeros(clean_size, dtype=int)
    })

    fraudulent_data = pd.DataFrame({
        'zip_code': generate_zip_codes(fraudulent_size),
        'transaction_amount': np.concatenate([np.random.exponential(scale=500, size=int(fraudulent_size * 0.5)),
                                              np.full(int(fraudulent_size * 0.5), np.random.randint(1000, 5000))]),
        'unix_time': np.concatenate([np.array([int((datetime.now() - timedelta(minutes=np.random.randint(1, 60))).timestamp()) for _ in range(int(fraudulent_size * 0.5))]),
                                     np.array([int((datetime.now() - timedelta(days=np.random.randint(1, 10))).timestamp()) for _ in range(int(fraudulent_size * 0.5))])]),
        'is_fraud': np.ones(fraudulent_size, dtype=int)
    })

    return pd.concat([clean_data, fraudulent_data], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Create the dataset
data = generate_dataset(1000)

# Prepare features and target
X = data[['zip_code', 'transaction_amount', 'unix_time']]
y = data['is_fraud']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a neural network
nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42, max_iter=500)
nn_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = nn_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1]))

filename = 'model_zip-amt-time.pkl'
with open(filename, 'wb') as file:
    pickle.dump(nn_model, file)

filename = 'scaler_zip-amt-time.pkl'
with open(filename, 'wb') as file:
    pickle.dump(scaler, file)