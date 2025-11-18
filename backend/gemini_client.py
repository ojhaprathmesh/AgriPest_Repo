import os
import json
import requests


def fetch_insect_info(name: str, harmful_label: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    prompt = (
        "You are an entomology assistant. Given an insect species name and whether it is harmful or harmless, "
        "return a concise JSON object with keys: family (string), habitat (array of strings), "
        "recommendation (array of short actionable handling steps), description (string). "
        "Only output JSON without extra text. Use scientifically credible facts. "
        f"Species: {name}. Harmfulness: {harmful_label}."
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512
        }
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            return None
        text = parts[0].get("text") or ""
        text = text.strip()
        try:
            obj = json.loads(text)
            fam = obj.get("family")
            hab = obj.get("habitat") if isinstance(obj.get("habitat"), list) else []
            rec = obj.get("recommendation") if isinstance(obj.get("recommendation"), list) else []
            desc = obj.get("description")
            if not fam or not desc:
                return None
            return {
                "family": fam,
                "habitat": hab,
                "recommendation": rec,
                "description": desc,
            }
        except Exception:
            return None
    except Exception:
        return None