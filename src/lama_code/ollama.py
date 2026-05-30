import http.client
import json
import urllib.parse
import urllib.request
from typing import Iterator


class OllamaError(Exception):
    pass


def list_models(base_url: str) -> list[str]:
    """Return sorted list of locally installed Ollama model names."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, messages: list[dict]) -> Iterator[str]:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": True,
        }).encode()

        parsed = urllib.parse.urlparse(f"{self.base_url}/api/chat")
        try:
            # Short timeout for TCP connection only — fail fast if Ollama is down
            conn = http.client.HTTPConnection(
                parsed.hostname, parsed.port or 80, timeout=10
            )
            conn.request(
                "POST", parsed.path, body=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            # Remove read timeout — TCP resets naturally if Ollama crashes
            conn.sock.settimeout(None)
        except (ConnectionRefusedError, OSError, TimeoutError) as e:
            raise OllamaError(
                f"Impossible de joindre Ollama sur {self.base_url}: {e}"
            ) from e

        try:
            for raw_line in resp:
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                if token := chunk.get("message", {}).get("content", ""):
                    yield token
                if chunk.get("done"):
                    break
        except (OSError, ConnectionResetError) as e:
            raise OllamaError(f"Ollama a fermé la connexion: {e}") from e
        finally:
            conn.close()
