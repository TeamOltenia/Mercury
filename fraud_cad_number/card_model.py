from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_spli
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import pickle

df = pd.read_csv("card_fraud.csv")

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
