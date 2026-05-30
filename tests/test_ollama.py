import json
import pytest
from unittest.mock import patch, MagicMock
from lama_code.ollama import OllamaClient, OllamaError


def _stream_lines(*tokens):
    lines = [
        json.dumps({"message": {"content": t}, "done": False}).encode()
        for t in tokens
    ]
    lines.append(json.dumps({"message": {"content": ""}, "done": True}).encode())
    return lines


def _mock_conn(tokens):
    """Build a mock http.client.HTTPConnection that streams tokens."""
    mock_resp = MagicMock()
    mock_resp.__iter__ = lambda s: iter(_stream_lines(*tokens))

    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp
    mock_conn.sock = MagicMock()
    return mock_conn


def test_generate_streams_tokens():
    client = OllamaClient("http://localhost:11434", "phi4-mini")
    mock_conn = _mock_conn(["Hello", " world"])

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        tokens = list(client.generate([{"role": "user", "content": "hi"}]))

    assert tokens == ["Hello", " world"]


def test_generate_raises_on_connection_error():
    client = OllamaClient("http://localhost:11434", "phi4-mini")
    with patch("http.client.HTTPConnection", side_effect=ConnectionRefusedError("refused")):
        with pytest.raises(OllamaError, match="Impossible de joindre Ollama"):
            list(client.generate([{"role": "user", "content": "hi"}]))
