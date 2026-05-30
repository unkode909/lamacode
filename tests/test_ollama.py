import json
import pytest
from unittest.mock import patch, MagicMock
from urllib.error import URLError
from lama_code.ollama import OllamaClient, OllamaError


def _stream_lines(*tokens):
    lines = [
        json.dumps({"message": {"content": t}, "done": False}).encode()
        for t in tokens
    ]
    lines.append(json.dumps({"message": {"content": ""}, "done": True}).encode())
    return lines


def test_generate_streams_tokens():
    client = OllamaClient("http://localhost:11434", "phi4-mini")
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.__iter__ = lambda s: iter(_stream_lines("Hello", " world"))

    with patch("urllib.request.urlopen", return_value=mock_resp):
        tokens = list(client.generate([{"role": "user", "content": "hi"}]))

    assert tokens == ["Hello", " world"]


def test_generate_raises_on_connection_error():
    client = OllamaClient("http://localhost:11434", "phi4-mini")
    with patch("urllib.request.urlopen", side_effect=URLError("refused")):
        with pytest.raises(OllamaError, match="Impossible de joindre Ollama"):
            list(client.generate([{"role": "user", "content": "hi"}]))
