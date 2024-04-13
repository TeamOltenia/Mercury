import httpx
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import requests
import json
import os
from openai import OpenAI
from starlette.responses import JSONResponse

sales_doc = "../doccuments/sales_and_eodStocks.xlsx"
transactions_doc = "../doccuments/transactions.xlsx"

sales_df = pd.read_excel(sales_doc)
df_csv = pd.read_csv('sales_data.csv')
sales_df['Product_ID'] = sales_df['Product_ID'].astype(str)
transactions_sales_df = pd.read_excel(transactions_doc)
print(sales_df.head())
from fastapi.encoders import jsonable_encoder
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)



# Step 4: Labeling Anomalies
def label_anomalies(row, too_much_stock_threshold, too_small_stock_threshold, not_enough_stock_threshold):
    if row['StockDiff'] > too_much_stock_threshold:
        return 'Too Much Stock'
    elif row['StockDiff'] < too_small_stock_threshold:
        return 'Too Small Stock'
    elif row['EndOfDayStock'] < not_enough_stock_threshold:
        return 'Not Enough Stock'
    else:
        return 'Normal'

app = FastAPI()
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


@app.get("/sales/{start_date}/{end_date}/{product_id}")
async def get_sales(start_date: str, end_date: str, product_id: str):
    product_data = sales_df[(sales_df['Product_ID'] == product_id) & (sales_df['Date'] >= start_date) & (sales_df['Date'] <= end_date)].copy()
    return product_data.to_json(orient="records", date_format="iso")


@app.get("/sales/anomaly/{start_date}/{end_date}/{product_id}")
async def get_anomaly_sales(start_date: str, end_date: str, product_id: str):
    product_data = sales_df[(sales_df['Product_ID'] == product_id) & (sales_df['Date'] >= start_date) & (sales_df['Date'] <= end_date)].copy()

    product_data = product_data.sort_values(by=['Product_ID', 'Date'])

    # Step 2: Feature Engineering
    # Calculate StockDiff as the difference in stock levels between consecutive days within the same product
    product_data['StockDiff'] = product_data.groupby('Product_ID')['EndOfDayStock'].diff()
    product_data = product_data.dropna(subset=['Product_ID', 'StockDiff', 'EndOfDayStock'])

    # Adjusted quantile values for "Normal" classification
    too_much_stock_threshold = product_data['EndOfDayStock'].quantile(0.95)
    too_small_stock_threshold = product_data['EndOfDayStock'].quantile(0.05)
    not_enough_stock_threshold = product_data['EndOfDayStock'].quantile(0.25)
    print(too_much_stock_threshold, too_small_stock_threshold, not_enough_stock_threshold)

    # Step 3: Define Anomalies
    product_data['Anomaly'] = product_data.apply(lambda row: label_anomalies(row, too_much_stock_threshold, too_small_stock_threshold, not_enough_stock_threshold), axis=1)

    # Step 5: Machine Learning Model (Isolation Forest)
    X = product_data[['EndOfDayStock', 'StockDiff']]  # Remove 'Date' column

    model = IsolationForest(contamination=0.05)
    model.fit(X)

    # Step 6: Anomaly Detection
    product_data['AnomalyDetection'] = model.predict(X)
    # Step 7: Visualization
    anomalies = product_data[product_data['AnomalyDetection'] == -1]
    print(anomalies)
    return anomalies.to_json(orient="records", date_format="iso")

MAILJET_API_KEY = "17f26b0d7fd61fe07a34cdeea7f1fbac"
MAILJET_API_SECRET = "14ea81b210dd1b904b9af8189474b2cd"
MAILJET_API_URL = "https://api.mailjet.com/v3.1/send"



@app.get("/sales/metrics/{start_date}/{end_date}/{product_id}")
async def get_sales(start_date: str, end_date: str, product_id: str):
    product_data = sales_df[(sales_df['Product_ID'] == product_id) & (sales_df['Date'] >= start_date) & (sales_df['Date'] <= end_date)].copy()
    if product_data.empty:
        return []
    total_sales = product_data['Sales'].sum()
    average_sales = product_data['Sales'].mean()
    total_revenue = product_data['Revenue'].sum()
    max_stock = product_data['EndOfDayStock'].min()
    sales_change_percentage = product_data['Sales'].pct_change().mean() * 100
    average_revenue_per_sale = total_revenue / total_sales

    metrics = {
        'Total Sales': float(total_sales),
        'Average Sales': float(average_sales),
        'Total Revenue': float(total_revenue),
        'Maximum End of Day Stock': float(max_stock),
        'Sales Change Percentage': float(sales_change_percentage),
        'Average Revenue per Sale': float(average_revenue_per_sale)
    }
    return json.dumps(metrics)

def predict_sales(product_id, date, model, label_encoder):
    # Convert date to datetime
    date = pd.to_datetime(date)

    # Extract year, month, and day
    year = date.year
    month = date.month
    day = date.day

    # Encode Product_ID
    encoded_product_id = label_encoder.transform([product_id])[0]

    # Create DataFrame for the prediction
    df_pred = pd.DataFrame({'Product_ID_Encoded': [encoded_product_id],
                            'Year': [year], 'Month': [month], 'Day': [day]})

    # Make prediction
    predicted_sales = model.predict(df_pred)[0]

    return predicted_sales




@app.get("/sales/predictions/{product_id_to_predict}/{year}/{start_month}/{end_month}")
def get_predictions(product_id_to_predict,year,start_month,end_month):

    df_csv['Date'] = pd.to_datetime(df_csv['Date'])


    # Extract year, month, and day from 'Date'
    df_csv['Year'] = df_csv['Date'].dt.year
    df_csv['Month'] = df_csv['Date'].dt.month
    df_csv['Day'] = df_csv['Date'].dt.day

    label_encoder = LabelEncoder()
    df_csv['Product_ID_Encoded'] = label_encoder.fit_transform(df_csv['Product_ID'].astype(str))

    # Select features and target
    X = df_csv[['Product_ID_Encoded', 'Year', 'Month', 'Day']]
    y = df_csv['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    predictions = []
    dates = []
    print(start_month, end_month, type(start_month))
    for month in range(int(start_month), int(end_month) + 1):
        # Number of days in the month
        num_days = pd.Period(f'{year}-{month}').days_in_month

        # Generate predictions every three days in the month
        for day in range(1, num_days + 1, 3):  # Increment by 3
            date_str = f'{year}-{month:02d}-{day:02d}'  # Format the date as YYYY-MM-DD
            prediction = predict_sales(product_id_to_predict, date_str, model, label_encoder)
            predictions.append(float(prediction))
            dates.append(date_str)

    return json.dumps({"dates" : dates, "predictions" : predictions})


def create_prompter( product_id, amount, language, customer, description, customer_email, increase='increase'):
    if customer:
        if increase:
            action = ""
        else:
            action = f"which has an decrese in price of the ammount of {amount} "
    else:
        action = "increase" if increase else "decrease"
    if customer:
        prompt = f"Create a short message which will be exported as JSON to tell a Client with email {customer_email} that based on his purchases he might be intersted in buying product with the product with this description: {description}, {action}%.The language of the message should be {language} :  Eg of an generated message in romanian : “Bună ziua Buyer Y,Având în vedere interesul dvs. anterior pentru Produsul Z, vă oferim un discount exclusiv de 15% valabil 48 de ore din momentul recepționării acestei notificări. Detalii suplimentare despre oferta noastră specială și despre produs pot fi găsite în aplicația mobilă RetailX, iar pentru comoditatea dvs., am trimis și un mesaj WhatsApp cu link direct către produs. De asemenea, puteți verifica emailul dvs. pentru mai multe informații"
    else:
        prompt = f"Create a short message which will be exported as JSON to tell a Shop Owner that his product with the product_Id of {product_id} has to {action} the stock of his product by {amount}%.The language of the message should be {language}:  Eg of an generated message in romanian : “Dragă Manager de Stoc X, Bazându-ne pe ultimele analize, vă sugerăm să ajustați stocul pentru Produsul Y. Recomandăm o creștere cu 20% a stocului, având în vedere tendințele de creștere ale cererii. Pentru a evita ruptura de stoc, sugerăm plasarea unei comenzi suplimentare în următoarele Z zile. Vă rugăm să verificați detaliile complete ale recomandării în dashboard-ul nostru de management al stocurilor. De asemenea, vă vom trimite un rezumat detaliat prin email și o alertă prin SMS pentru a asigura o acțiune promptă."
    return prompt


def query_openai(prompt, engine='gpt-3.5-turbo', max_tokens=100):
    api_key = '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'  # Replace with your actual API key
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a text generator which generates short messages (max: 500 characters) for the users based on their request"},
            {"role": "user", "content": prompt}
        ]
    )
    return response

def recommendation(customer_id, num_recommendations=5):
    df = transactions_sales_df.copy()
    df.dropna(inplace=True)

    # Ensure correct data types (e.g., Customer_Id as string)
    df['Customer ID'] = df['Customer ID'].astype(str)

    # Normalize 'Price' column
    scaler = MinMaxScaler()
    df['Price'] = scaler.fit_transform(df[['Price']])

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA

    # Extract features from 'Description'
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform(df['Description'])

    # Reduce dimensions (optional, for better performance)
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X.toarray())


    from sklearn.cluster import KMeans

    # Cluster for category identification
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X_reduced)
    df['category'] = kmeans.labels_


    from sklearn.cluster import KMeans

    # Aici modifica numarul de clusetere daca vrei sa ai mai multe categori posivilie penntru obiecte
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X_reduced)
    df['category'] = kmeans.labels_

    # Create user profiles by aggregating data
    user_profile = df.groupby('Customer ID').agg({
        'category': lambda x: x.mode()[0],  # Most common category
        'Price': 'mean',  # Average spending
    })

    if customer_id not in user_profile.index:
        print(f"Customer ID {customer_id} not found.")
        return None

    user_cat = user_profile.loc[customer_id, 'category']
    similar_products = df[df['category'] == user_cat]

    # Recommend products from the same category
    recommendations = similar_products.sample(n=num_recommendations)
    return recommendations['Product_ID']

def get_product_name(id):
    product_name = transactions_sales_df.loc[transactions_sales_df['Product_ID'] == id, 'Description'].iloc[0]
    return product_name

@app.get("/recommendation/{customer_id}")
def get_product_recommendation(customer_id):
    return recommendation(customer_id)


@app.get("/recommendation/customer/{customer_id}/{customer_email}/{language}")
def generate_emails_customer(customer_id, customer_email, language):
    
    messages = []
    recs = recommendation(customer_id)
    for key, value in recs.items():
        print(value)
        description = get_product_name(value)
        print(description)
        prompt = create_prompter(customer_id, 20, language, True, description, customer_email, False)
        print(prompt)
        response = query_openai(prompt)
        print(response)
        if response:
            messages.append(response.choices[0].message.content)

@app.get("/recommendation/customer/{customer_id}/{customer_email}/{language/{endDate}/{startDate}")
def generate_emails_manager(customer_id, customer_email,startDate,endDate,language='english'):

    product_data = sales_df[(sales_df['Product_ID'] == customer_id) & (sales_df['Date'] >= startDate) & (
                sales_df['Date'] <= endDate)].copy()

    product_data = product_data.sort_values(by=['Product_ID', 'Date'])

    # Step 2: Feature Engineering
    # Calculate StockDiff as the difference in stock levels between consecutive days within the same product
    product_data['StockDiff'] = product_data.groupby('Product_ID')['EndOfDayStock'].diff()
    product_data = product_data.dropna(subset=['Product_ID', 'StockDiff', 'EndOfDayStock'])

    # Adjusted quantile values for "Normal" classification
    too_much_stock_threshold = product_data['EndOfDayStock'].quantile(0.95)
    too_small_stock_threshold = product_data['EndOfDayStock'].quantile(0.05)
    not_enough_stock_threshold = product_data['EndOfDayStock'].quantile(0.25)
    print(too_much_stock_threshold, too_small_stock_threshold, not_enough_stock_threshold)

    # Step 3: Define Anomalies
    product_data['Anomaly'] = product_data.apply(
        lambda row: label_anomalies(row, too_much_stock_threshold, too_small_stock_threshold,
                                    not_enough_stock_threshold), axis=1)

    # Step 5: Machine Learning Model (Isolation Forest)
    X = product_data[['EndOfDayStock', 'StockDiff']]  # Remove 'Date' column

    model = IsolationForest(contamination=0.05)
    model.fit(X)

    # Step 6: Anomaly Detection
    product_data['AnomalyDetection'] = model.predict(X)
    # Step 7: Visualization
    anomalies = product_data[product_data['AnomalyDetection'] == -1]

    # if anomalies.get(anomalies.size-1)

    messages = []
    recs = recommendation(customer_id)
    for key, value in recs.items():
        print(value)
        description = get_product_name(value)
        print(description)
        prompt = create_prompter(customer_id, 20, language, False, description, customer_email, False)
        print(prompt)
        response = query_openai(prompt)
        print(response)
        if response:
            messages.append(response.choices[0].message.content)

    return messages


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)