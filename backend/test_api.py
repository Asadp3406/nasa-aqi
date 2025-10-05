import requests
import json

# Test the API
url = "http://localhost:5000/api/forecast"
data = {"city": "New Delhi, India"}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}...")
except Exception as e:
    print(f"Error: {e}")