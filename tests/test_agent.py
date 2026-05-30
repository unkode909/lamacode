import pytest
from unittest.mock import MagicMock
from lama_code.agent import parse_bash_blocks, Agent
from lama_code.config import Config
from lama_code.executor import ExecutionResult


def test_parse_no_blocks():
    assert parse_bash_blocks("Juste du texte.") == []


def test_parse_one_block():
    assert parse_bash_blocks("Voici :\n```bash\nls -la\n```") == ["ls -la"]


def test_parse_multiple_blocks():
    text = "```bash\necho a\n```\nEt aussi :\n```bash\necho b\n```"
    assert parse_bash_blocks(text) == ["echo a", "echo b"]


def test_parse_ignores_non_bash():
    assert parse_bash_blocks("```python\nprint('hi')\n```") == []


def _make_agent(responses, yolo=True):
    cfg = Config(yolo=yolo, max_cycles=5, context_window=25)
    ollama = MagicMock()
    call_count = [0]

    def generate(messages):
        idx = call_count[0]
        call_count[0] += 1
        tokens = responses[idx] if idx < len(responses) else ["Terminé."]
        return iter(tokens)

    ollama.generate.side_effect = generate
    display = MagicMock()
    display.confirm.return_value = True

    def fake_execute(cmd):
        return ExecutionResult(command=cmd, stdout=f"output:{cmd}", stderr="", exit_code=0)

    return Agent(cfg=cfg, ollama=ollama, display=display, execute_fn=fake_execute)


def test_run_text_only_returns_response():
    agent = _make_agent([["Bonjour !"]])
    assert agent.run("dis bonjour") == "Bonjour !"


def test_run_with_block_loops_then_returns():
    agent = _make_agent([
        ["Voici :\n```bash\necho hi\n```"],
        ["La commande a affiché hi."],
    ])
    result = agent.run("dis hi")
    assert result == "La commande a affiché hi."


def test_run_respects_max_cycles():
    agent = _make_agent([["```bash\necho x\n```"] for _ in range(10)])
    agent.cfg.max_cycles = 3
    result = agent.run("boucle infinie")
    assert "Max cycles" in result


def test_history_grows():
    agent = _make_agent([["Réponse."]])
    agent.run("Question")
    assert len(agent.history) == 2
