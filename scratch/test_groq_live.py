import os, requests, json
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get('GROQ_API_KEY')
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
payload = {
    'model': 'openai/gpt-oss-120b',
    'messages': [
        {'role': 'system', 'content': 'Output valid JSON: {"intent": "string", "confidence": 0.95}'},
        {'role': 'user', 'content': 'My battery is dying fast'}
    ],
    'response_format': {'type': 'json_object'},
    'max_tokens': 300
}
r = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload, timeout=10)
print('Status:', r.status_code)
print('Response JSON:', r.json())
