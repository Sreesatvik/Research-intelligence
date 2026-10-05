import json
import os
import time
from typing import Iterator

import requests
from dotenv import load_dotenv

load_dotenv()  # reads key=value lines from .env into os.environ

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_WAIT_SECONDS = 30  # longer waits mean a quota problem, not a brief rate limit


class GroqLLM:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        self.tokens_used = 0  # running total, for cost awareness
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment")

    def _open(self, body: dict, stream: bool = False) -> requests.Response:
        """Send the request, waiting out brief rate limits. Raises with the server's explanation on failure."""
        for attempt in range(6):
            response = requests.post(GROQ_URL, json=body, timeout=60, stream=stream,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code != 429:
                break
            wait = float(response.headers.get("retry-after", 2 ** (attempt + 1)))
            if wait > MAX_WAIT_SECONDS:  # e.g. the DAILY quota is used up: do not hang, say so
                break
            time.sleep(wait)
        if not response.ok:  # include the server's explanation, not just the status code
            raise requests.HTTPError(f"{response.status_code}: {response.text[:500]}", response=response)
        return response

    def _post(self, body: dict) -> dict:
        data = self._open(body).json()
        self.tokens_used += data.get("usage", {}).get("total_tokens", 0)
        return data

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.chat(messages, temperature=temperature)["content"]

    def stream(self, system: str, user: str, temperature: float = 0.0) -> Iterator[str]:
        """Yield the answer in pieces as the model writes it, so a reader sees words straight away."""
        body = {"model": self.model, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        response = self._open(body, stream=True)
        try:
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                usage = chunk.get("usage") or chunk.get("x_groq", {}).get("usage")
                if usage:  # sent once, on the last chunk
                    self.tokens_used += usage.get("total_tokens", 0)
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
        finally:
            response.close()  # also runs when the reader stops early

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.0) -> dict:
        """Send a full message list (optionally with tool definitions); return the reply message."""
        body = {"model": self.model, "temperature": temperature, "messages": messages}
        if tools:
            body["tools"] = tools
        return self._post(body)["choices"][0]["message"]
