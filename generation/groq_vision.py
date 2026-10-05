import base64
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqVLM:
    def __init__(self, model: str = "qwen/qwen3.8-27b"):
        self.model = model
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment")

    def ask_image(self, image_png: bytes, question: str) -> str:
        b64 = base64.b64encode(image_png).decode()
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 600,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        }
        for attempt in range(6):
            response = requests.post(GROQ_URL, json=body, timeout=120,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code != 429:
                break
            time.sleep(float(response.headers.get("retry-after", 2 ** (attempt + 1))))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
