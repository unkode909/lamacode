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
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    # Special prefix used to distinguish thinking tokens from content tokens
    THINK_PREFIX = "\x00think\x00"

    # Models that support chain-of-thought thinking
    _THINKING_MODELS = ("deepseek-r1", "deepseek-r2", "qwq", "phi4-reasoning")

    def _supports_thinking(self) -> bool:
        return any(name in self.model.lower() for name in self._THINKING_MODELS)

    def generate(self, messages: list[dict], stats: dict | None = None) -> Iterator[str]:
        payload_dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self._supports_thinking():
            payload_dict["think"] = True
        payload = json.dumps(payload_dict).encode()

        parsed = urllib.parse.urlparse(f"{self.base_url}/api/chat")
        try:
            # Short timeout for TCP connection only — fail fast if Ollama is down
            conn = http.client.HTTPConnection(
                parsed.hostname, parsed.port or 80, timeout=10
            )
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            conn.request("POST", parsed.path, body=payload, headers=headers)
            # Remove timeout before getresponse() — model loading can take >10s
            # TCP resets naturally if Ollama crashes mid-generation
            conn.sock.settimeout(None)
            resp = conn.getresponse()
        except (ConnectionRefusedError, OSError, TimeoutError) as e:
            raise OllamaError(
                f"Impossible de joindre Ollama sur {self.base_url}: {e}"
            ) from e

        try:
            for raw_line in resp:
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                msg = chunk.get("message", {})
                if think := msg.get("thinking", ""):
                    yield self.THINK_PREFIX + think
                if token := msg.get("content", ""):
                    yield token
                if chunk.get("done"):
                    if stats is not None:
                        stats["prompt_tokens"] = chunk.get("prompt_eval_count", 0)
                        stats["generated_tokens"] = chunk.get("eval_count", 0)
                    break
        except (OSError, ConnectionResetError) as e:
            raise OllamaError(f"Ollama a fermé la connexion: {e}") from e
        finally:
            conn.close()
