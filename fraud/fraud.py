import datetime

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



train_data = pd.read_csv("fraudTrain.csv")
print(train_data.info)

train_data.drop(columns=['Unnamed: 0','merch_lat','merch_long', 'dob','gender','lat','long','city_pop','job', 'trans_num','trans_date_trans_time'],inplace=True)
train_data.dropna(ignore_index=True)

categorical_columns = ['merchant', 'category', 'first', 'last', 'street', 'city', 'state']
encoders = {}
for col in categorical_columns:
    encoder = LabelEncoder()
    train_data[col] = encoder.fit_transform(train_data[col])
    encoders[col] = encoder
    joblib.dump(encoder, f'{col}_encoder.pkl')

X = train_data.drop(columns=["is_fraud"], inplace = False)
Y = train_data["is_fraud"]

model = SVC()
model.fit(X, Y)

print(model.score(X, Y))



test_data = pd.read_csv("fraudTest.csv")

test_data.drop(columns=['Unnamed: 0','merch_lat','merch_long', 'dob','gender','lat','long','city_pop','job', 'trans_num','trans_date_trans_time'],inplace=True)
test_data.dropna(ignore_index=True)

test_data["merchant"] = encoder.fit_transform(test_data["merchant"])
test_data["category"] = encoder.fit_transform(test_data["category"])
test_data["first"] = encoder.fit_transform(test_data["first"])
test_data["last"] = encoder.fit_transform(test_data["last"])
test_data["street"] = encoder.fit_transform(test_data["street"])
test_data["city"] = encoder.fit_transform(test_data["city"])
test_data["state"] = encoder.fit_transform(test_data["state"])

X_test = test_data.drop(columns=["is_fraud"], inplace = False)
Y_test = test_data["is_fraud"]


X_test = test_data.drop(columns=["is_fraud"], inplace = False)
Y_test = test_data["is_fraud"]

y_pred = model.predict(X_test)

accuracy = accuracy_score(test_data['is_fraud'],y_pred)
print(accuracy)

filename = 'finalized2_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
