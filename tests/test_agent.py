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

    def generate(messages, stats=None):
        idx = call_count[0]
        call_count[0] += 1
        tokens = responses[idx] if idx < len(responses) else ["Terminé."]
        return iter(tokens)

    ollama.generate.side_effect = generate
    display = MagicMock()
    display.confirm.return_value = True

    def fake_execute(cmd, on_stdout=None, on_stderr=None, on_stdin_needed=None):
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


def test_confirm_skips_block_when_denied():
    """Verifies the non-yolo confirm path."""
    cfg = Config(yolo=False, max_cycles=5, context_window=25)
    ollama = MagicMock()
    ollama.generate.return_value = iter(["Réponse finale."])
    display = MagicMock()
    display.confirm.return_value = False  # user denies

    # First call: bash block; second call (after denied result): text only
    responses = [
        ["```bash\necho hi\n```"],
        ["La commande a été ignorée."],
    ]
    call_count = [0]
    def generate(messages, stats=None):
        idx = call_count[0]
        call_count[0] += 1
        return iter(responses[idx] if idx < len(responses) else ["Terminé."])
    ollama.generate.side_effect = generate

    def fake_execute(cmd, on_stdout=None, on_stderr=None, on_stdin_needed=None):
        return ExecutionResult(command=cmd, stdout="output", stderr="", exit_code=0)

    agent = Agent(cfg=cfg, ollama=ollama, display=display, execute_fn=fake_execute)
    result = agent.run("dis hi")

    # confirm was called, execute was NOT called (block skipped)
    display.confirm.assert_called_once()
    assert "[ignoré par l'utilisateur]" in agent.history[2].content


def test_build_messages_includes_system_prompt():
    """Verifies user system_prompt is prepended before TOOL_INSTRUCTIONS."""
    from lama_code.agent import TOOL_INSTRUCTIONS
    cfg = Config(system_prompt="Sois concis.", max_cycles=5, context_window=25)
    ollama = MagicMock()
    ollama.generate.side_effect = lambda messages, stats=None: iter(["ok"])
    display = MagicMock()
    display.confirm.return_value = True

    agent = Agent(cfg=cfg, ollama=ollama, display=display,
                  execute_fn=lambda cmd, **kw: ExecutionResult(cmd, "", "", 0))
    agent.run("test")

    # First message sent to ollama should be the system message
    call_args = ollama.generate.call_args[0][0]
    system_msg = call_args[0]
    assert system_msg["role"] == "system"
    assert "Sois concis." in system_msg["content"]
    assert TOOL_INSTRUCTIONS in system_msg["content"]
    # System prompt comes before TOOL_INSTRUCTIONS
    assert system_msg["content"].index("Sois concis.") < system_msg["content"].index(TOOL_INSTRUCTIONS)


def test_format_result_truncated():
    cfg = Config(yolo=True, max_cycles=5, context_window=25)
    ollama = MagicMock()
    responses = [
        ["```bash\nseq 1 300\n```"],
        ["ok"],
    ]
    call_count = [0]
    def generate(messages, stats=None):
        idx = call_count[0]
        call_count[0] += 1
        return iter(responses[idx] if idx < len(responses) else ["ok"])
    ollama.generate.side_effect = generate
    display = MagicMock()
    display.confirm.return_value = True

    truncated_result = ExecutionResult(
        command="seq 1 300",
        stdout="1\n2\n3\n[... 297 lignes supplémentaires dans /tmp/lama-abc.txt]",
        stderr="",
        exit_code=0,
        output_file="/tmp/lama-abc.txt",
        truncated=True,
        total_lines=300,
    )

    def fake_execute(cmd, on_stdout=None, on_stderr=None, on_stdin_needed=None):
        return truncated_result

    agent = Agent(cfg=cfg, ollama=ollama, display=display, execute_fn=fake_execute)
    agent.run("compte jusqu'a 300")

    result_msg = agent.history[2].content
    assert "/tmp/lama-abc.txt" in result_msg


def test_stdin_needed_ai_responds():
    cfg = Config(yolo=True, max_cycles=5, context_window=25)
    ollama = MagicMock()
    responses = [
        ["```bash\nread -p 'Name: ' n\n```"],
        ["```stdin\nAlice\n```"],
        ["Réponse finale."],
    ]
    call_count = [0]
    def generate(messages, stats=None):
        idx = call_count[0]
        call_count[0] += 1
        return iter(responses[idx] if idx < len(responses) else ["ok"])
    ollama.generate.side_effect = generate

    display = MagicMock()
    display.confirm.return_value = True
    stdin_injected = []

    def fake_execute(cmd, on_stdout=None, on_stderr=None, on_stdin_needed=None):
        if on_stdin_needed:
            val = on_stdin_needed("Name: ")
            stdin_injected.append(val)
        return ExecutionResult(command=cmd, stdout="Name: Alice\n", stderr="", exit_code=0)

    agent = Agent(cfg=cfg, ollama=ollama, display=display, execute_fn=fake_execute)
    agent.run("dis mon nom")

    assert stdin_injected[0] == "Alice\n"
