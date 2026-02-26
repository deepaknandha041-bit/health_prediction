import requests
import json

def verify_api():
    url = "http://127.0.0.1:5000/api"
    payload = {
        "fever": 1,
        "headache": 0,
        "cough": 1,
        "fatigue": 1,
        "vomiting": 0,
        "cold": 1
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            print("API Verification Successful!")
            print("Response:", response.json())
        else:
            print(f"API Verification Failed with status code: {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print(f"An error occurred during verification: {e}")

if __name__ == "__main__":
    verify_api()
