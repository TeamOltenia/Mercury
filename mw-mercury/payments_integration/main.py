import json
import urllib3

FRAUD_DETENCTING_ENDPOINT = ""
LOGGING_LAMBDA = ""

def stripe_forwarding(payment_details):
    return {"status": 200, "message": "Your stripe transaction was successfully processed"}

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
    return True
    response = requests.post(FRAUD_DETENCTING_ENDPOINT, json=payment_details).json()
    if response["status"] == 200:
        return False
    return True


def logging(payload):

    return {
        'statusCode': 200,
        'body': json.dumps('Success!')
    }

    response = requests.post("LOGGING_LAMBDA", json = payload)
    if response.status_code == 200:
        return {
        'statusCode': 200,
        'body': json.dumps('Success!')
    }
    return {
        'statusCode': 400,
        'body': response.json()

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

    if is_fraud(payment_details):
        return_message = payment_provider_routing(payment_provider, payment_details)
    else:
        return_message = {"status": 406, "message": "This tranzactions is fraudulent"}

    response = logging(return_message)

    