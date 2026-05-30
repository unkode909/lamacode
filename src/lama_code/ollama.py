import json
import urllib.request
import urllib.error
from typing import Iterator


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, messages: list[dict]) -> Iterator[str]:
        url = f"{self.base_url}/api/chat"
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except (urllib.error.URLError, TimeoutError) as e:
            raise OllamaError(
                f"Impossible de joindre Ollama sur {self.base_url}: {e}"
            ) from e

        with resp:
            for line in resp:
                if not line:
                    continue
                chunk = json.loads(line)
                if token := chunk.get("message", {}).get("content", ""):
                    yield token
                if chunk.get("done"):
                    break
