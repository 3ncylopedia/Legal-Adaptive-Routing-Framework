## Saint Louis University
## Team 404FoundUs
## @file tests/test_emb.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Multi-model embedding endpoint validation test.

import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# models to test
models = [
    "sentence-transformers/all-minilm-l6-v2",
    "jinaai/jina-embeddings-v2-base-en",
    "cohere/embed-english-v3.0",
    "openai/text-embedding-3-small"
]

## @iter_ models : Validating each embedding model endpoint
for m in models:
    payload = {
        "model": m,
        "input": ["test sentence"]
    }
    
    resp = requests.post("https://openrouter.ai/api/v1/embeddings", headers=headers, json=payload)
    print(f"Model: {m}")
    try:
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Response: {list(data.keys())} | Errors: {data.get('error', None)}")
    except Exception as e:
        print("Error parsing", e, resp.text)
    print("-" * 30)
