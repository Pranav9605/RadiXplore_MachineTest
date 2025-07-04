import requests

def call_gemini(prompt: str, api_key: str) -> str:
    """
    Calls Google Gemini API (Generative Language) and returns the text content.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    resp = requests.post(f"{url}?key={api_key}", headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
