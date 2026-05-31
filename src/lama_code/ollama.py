import http.client
import json
import ssl
import urllib.parse
import urllib.request
from typing import Iterator


class OllamaError(Exception):
    pass


def list_models(base_url: str, api_key: str = "") -> list[str]:
    """Return sorted list of model names from local Ollama or cloud API."""
    base = base_url.rstrip("/")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Try /api/tags (local Ollama)
    for path in ("/api/tags", "/v1/models"):
        try:
            req = urllib.request.Request(f"{base}{path}", headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            # Local format: {"models": [{"name": ...}]}
            if "models" in data:
                return sorted(m["name"] for m in data["models"])
            # OpenAI format: {"data": [{"id": ...}]}
            if "data" in data:
                return sorted(m["id"] for m in data["data"])
        except Exception:
            continue
    return []


class OllamaClient:
    THINK_PREFIX = "\x00think\x00"
    _THINKING_MODELS = ("deepseek-r1", "deepseek-r2", "qwq", "phi4-reasoning")

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._is_cloud = "api.ollama.com" in self.base_url

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
        is_https = parsed.scheme == "https"

        try:
            if is_https:
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(
                    parsed.hostname, parsed.port or 443, timeout=10, context=ctx
                )
            else:
                conn = http.client.HTTPConnection(
                    parsed.hostname, parsed.port or 80, timeout=10
                )
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            conn.request("POST", parsed.path, body=payload, headers=headers)
            conn.sock.settimeout(None)
            resp = conn.getresponse()
        except (ConnectionRefusedError, OSError, TimeoutError) as e:
            raise OllamaError(
                f"Impossible de joindre Ollama sur {self.base_url}: {e}"
            ) from e

        try:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                # OpenAI-compatible SSE format: "data: {...}" or "data: [DONE]"
                if line.startswith(b"data: "):
                    data = line[6:]
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    # OpenAI format: choices[0].delta.content
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if token := delta.get("content", ""):
                        yield token
                    if chunk.get("choices", [{}])[0].get("finish_reason"):
                        if stats is not None:
                            usage = chunk.get("usage", {})
                            stats["prompt_tokens"] = usage.get("prompt_tokens", 0)
                            stats["generated_tokens"] = usage.get("completion_tokens", 0)
                        break
                else:
                    # Native Ollama format
                    chunk = json.loads(line)
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
