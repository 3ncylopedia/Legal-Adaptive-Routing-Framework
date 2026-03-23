import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "sentence-transformers/all-minilm-l6-v2",
    "input": ["test sentence " + str(i) for i in range(100)]
}

resp = requests.post("https://openrouter.ai/api/v1/embeddings", headers=headers, json=payload)
print(f"Status: {resp.status_code}")
data = resp.json()
if 'data' in data:
    print(f"Data length: {len(data['data'])}")
else:
    print(f"No data field. Keys: {data.keys()}")
    print(data)
