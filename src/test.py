# Run this in Python directly
import requests
response = requests.post("http://127.0.0.1:8000/api/v1/ai/populate-vector-store")
print(response.json())