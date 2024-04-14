import json
import requests  # Import the requests library
from time import sleep

FRAUD_DETECTING_ENDPOINT = "http://example.com/fraud_detection"
LOGGING_LAMBDA = "https://cocupv36mxfosdydthygmc3rha0kqzbx.lambda-url.eu-north-1.on.aws/"

def stripe_forwarding(payment_details):
    # stripe call
    return {"status": 200, "message": "Your stripe transaction was successfully processed and logs inserted"}

def payment_provider_routing(payment_provider: str, payment_details):
    match payment_provider.strip().lower():
        case "stripe":
            return stripe_forwarding(payment_details)
        case "paypal":
            pass
        case "amazon pay":
            pass
        case "klarna":
            pass

def is_fraud(payment_details) -> bool:
    response = requests.post(FRAUD_DETECTING_ENDPOINT, json=payment_details)
    if response.status_code == 200:
        response_data = response.json()
        return response_data["status"] != 200
    return True  # Default to True if status code is not 200

def logging(payload):
    try:
        response = requests.post(LOGGING_LAMBDA, json=payload)
        response_data = response.json()
        print("Logging response:", response_data)  # Print the response data for debugging
        print("Response Status Code:", response.status_code)  # Check and log the status code
    except Exception as e:
        print("Logging error:", str(e))  # Print any errors that occur
        if response:  # Check if response is available and print more debug information
            print("Response status:", response.status_code)
            print("Response text:", response.text)
        return {
            'statusCode': 500,
            'body': json.dumps("Invalid response format from logging service")
        }

def lambda_handler(event, context):
    try:
        request_body = json.loads(event.get('body', ''))
        payment_provider = request_body['payment_provider']
        payment_details = request_body['payment_details']
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }

    try:
        pre_payment_log = {"table_name": "Logs_Before", "request_body": payment_details}
        logging(pre_payment_log)
        if is_fraud(payment_details):
            payment_details["is_fraud"] = True
            return_message = payment_provider_routing(payment_provider, payment_details)
        else:
            payment_details["is_fraud"] = True

        post_payment_log = {"table_name": "Logs_After", "request_body": payment_details}
        sleep(2)
        logging(post_payment_log)

        return {
            'statusCode': 200,
            'body': json.dumps('Success!')
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
