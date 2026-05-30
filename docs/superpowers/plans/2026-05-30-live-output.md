# Live Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace blocking `subprocess.run` with live streaming execution — stdout in white, stderr in yellow, full output saved to `/tmp`, AI can inject stdin when a command waits for input.

**Architecture:** `execute_streaming()` in `executor.py` uses `Popen` + two reader threads (stdout/stderr) + a timeout-based stdin detector. `agent.py` provides callbacks and a new `_handle_stdin_needed()` method that consults the AI mid-execution. `display.py` gets four new output functions. `execute_fn` signature changes to accept keyword callbacks; existing tests are updated accordingly.

**Tech Stack:** Python stdlib (`subprocess`, `threading`, `tempfile`, `signal`), `rich` (already installed)

---

## File Map

| File | Change |
|------|--------|
| `src/lama_code/executor.py` | Add `execute_streaming()`, extend `ExecutionResult` with `output_file`, `truncated`, `total_lines` |
| `src/lama_code/display.py` | Add `stream_stdout_line()`, `stream_stderr_line()`, `show_stdin_waiting()`, `show_truncation()`; update `show_result()` |
| `src/lama_code/agent.py` | Update `execute_fn` type, rewrite `_execute_blocks()`, add `_handle_stdin_needed()`, add `_format_result()` |
| `src/lama_code/config.py` | Add `stdin_timeout: float = 3.0` and `max_output_lines: int = 200` to `Config` |
| `tests/test_executor.py` | Add tests for `execute_streaming()` |
| `tests/test_agent.py` | Update `fake_execute` signature in helpers; add streaming integration tests |

---

### Task 1: Extend Config with streaming parameters

**Files:**
- Modify: `src/lama_code/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
def test_streaming_defaults(tmp_path):
    with patch("lama_code.config._discover_ollama_url", return_value=DEFAULT_OLLAMA_URL):
        cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project")
    assert cfg.stdin_timeout == 3.0
    assert cfg.max_output_lines == 200


def test_streaming_params_from_file(tmp_path):
    (tmp_path / ".lama.md").write_text(
        "---\nstdin_timeout: 5.0\nmax_output_lines: 100\n---\n"
    )
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project")
    assert cfg.stdin_timeout == 5.0
    assert cfg.max_output_lines == 100
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_config.py::test_streaming_defaults tests/test_config.py::test_streaming_params_from_file -v
```

Expected: `AttributeError: 'Config' object has no attribute 'stdin_timeout'`

- [ ] **Step 3: Add fields to Config dataclass in `src/lama_code/config.py`**

In the `Config` dataclass, add after `system_prompt`:

```python
    stdin_timeout: float = 3.0
    max_output_lines: int = 200
```

In `load_config()`, in the overrides block after `max_cycles`, add:

```python
    if "stdin_timeout" in merged:
        overrides["stdin_timeout"] = float(merged["stdin_timeout"])
    if "max_output_lines" in merged:
        overrides["max_output_lines"] = int(merged["max_output_lines"])
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_config.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lama_code/config.py tests/test_config.py
git commit -m "feat: config — add stdin_timeout and max_output_lines"
```

---

### Task 2: Extend ExecutionResult and implement execute_streaming()

**Files:**
- Modify: `src/lama_code/executor.py`
- Modify: `tests/test_executor.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_executor.py`:

```python
from lama_code.executor import execute, execute_streaming


def test_streaming_stdout_live():
    lines = []
    result = execute_streaming(
        "printf 'line1\nline2\nline3\n'",
        on_stdout=lines.append,
        on_stderr=lambda l: None,
        on_stdin_needed=lambda _: None,
    )
    assert result.success is True
    assert result.exit_code == 0
    assert any("line1" in l for l in lines)
    assert any("line2" in l for l in lines)


def test_streaming_stderr_separate():
    stdout_lines = []
    stderr_lines = []
    result = execute_streaming(
        "echo out; echo err >&2",
        on_stdout=stdout_lines.append,
        on_stderr=stderr_lines.append,
        on_stdin_needed=lambda _: None,
    )
    assert any("out" in l for l in stdout_lines)
    assert any("err" in l for l in stderr_lines)


def test_streaming_output_file_created():
    result = execute_streaming(
        "echo hello",
        on_stdout=lambda l: None,
        on_stderr=lambda l: None,
        on_stdin_needed=lambda _: None,
    )
    import os
    assert result.output_file != ""
    assert os.path.exists(result.output_file)
    content = open(result.output_file).read()
    assert "hello" in content


def test_streaming_truncation():
    result = execute_streaming(
        "seq 1 300",
        on_stdout=lambda l: None,
        on_stderr=lambda l: None,
        on_stdin_needed=lambda _: None,
        max_output_lines=50,
    )
    assert result.truncated is True
    assert result.total_lines == 300
    # stdout sent to AI is truncated
    assert result.stdout.count("\n") <= 51  # 50 lines + truncation message


def test_streaming_result_has_new_fields():
    result = execute_streaming(
        "echo hi",
        on_stdout=lambda l: None,
        on_stderr=lambda l: None,
        on_stdin_needed=lambda _: None,
    )
    assert hasattr(result, "output_file")
    assert hasattr(result, "truncated")
    assert hasattr(result, "total_lines")
    assert result.truncated is False


def test_streaming_failing_command():
    result = execute_streaming(
        "ls /nonexistent_xyz_path",
        on_stdout=lambda l: None,
        on_stderr=lambda l: None,
        on_stdin_needed=lambda _: None,
    )
    assert result.success is False
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_executor.py::test_streaming_stdout_live -v
```

Expected: `ImportError: cannot import name 'execute_streaming'`

- [ ] **Step 3: Extend ExecutionResult in `src/lama_code/executor.py`**

Replace the `ExecutionResult` dataclass:

```python
@dataclass
class ExecutionResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    output_file: str = ""
    truncated: bool = False
    total_lines: int = 0

    @property
    def success(self) -> bool:
        return self.exit_code == 0
```

- [ ] **Step 4: Implement execute_streaming() in `src/lama_code/executor.py`**

Add after the existing `execute()` function:

```python
import os
import signal
import tempfile
import threading
import time
from typing import Callable


def execute_streaming(
    command: str,
    on_stdout: Callable[[str], None],
    on_stderr: Callable[[str], None],
    on_stdin_needed: Callable[[str], str | None],
    stdin_timeout: float = 3.0,
    max_output_lines: int = 200,
) -> ExecutionResult:
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    all_lines: list[str] = []
    last_line_time: list[float] = [time.time()]
    lock = threading.Lock()

    tmp = tempfile.NamedTemporaryFile(
        mode="w", prefix="lama-", suffix=".txt", delete=False
    )

    def record(line: str, kind: str) -> None:
        with lock:
            tmp.write(line)
            tmp.flush()
            all_lines.append(line)
            last_line_time[0] = time.time()
        if kind == "out":
            stdout_lines.append(line)
        else:
            stderr_lines.append(line)

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        tmp.close()
        return ExecutionResult(
            command=command, stdout="", stderr=str(e), exit_code=1,
            output_file=tmp.name,
        )

    def read_stream(stream, kind: str, cb: Callable[[str], None]) -> None:
        try:
            for line in stream:
                record(line, kind)
                cb(line)
        except Exception:
            pass

    t_out = threading.Thread(
        target=read_stream, args=(proc.stdout, "out", on_stdout), daemon=True
    )
    t_err = threading.Thread(
        target=read_stream, args=(proc.stderr, "err", on_stderr), daemon=True
    )
    t_out.start()
    t_err.start()

    stdin_requested = False
    try:
        while proc.poll() is None:
            time.sleep(0.2)
            if not stdin_requested:
                idle = time.time() - last_line_time[0]
                if idle >= stdin_timeout:
                    stdin_requested = True
                    current = "".join(all_lines)
                    value = on_stdin_needed(current)
                    if value is not None:
                        try:
                            proc.stdin.write(value)
                            proc.stdin.flush()
                        except Exception:
                            pass
                    else:
                        try:
                            user_input = input() + "\n"
                            proc.stdin.write(user_input)
                            proc.stdin.flush()
                        except Exception:
                            pass
                    stdin_requested = False  # reset — command may ask again
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    try:
        proc.stdin.close()
    except Exception:
        pass

    tmp.close()
    exit_code = proc.returncode if proc.returncode is not None else 1
    total = len(all_lines)

    if total > max_output_lines:
        kept = "".join(stdout_lines[:max_output_lines])
        truncated_msg = (
            f"\n[... {total - max_output_lines} lignes supplémentaires dans "
            f"{tmp.name}\n"
            f"Lis des sections avec : head -n 50 {tmp.name}\n"
            f"                        sed -n '200,250p' {tmp.name}]"
        )
        stdout_for_ai = kept + truncated_msg
        return ExecutionResult(
            command=command,
            stdout=stdout_for_ai,
            stderr="".join(stderr_lines),
            exit_code=exit_code,
            output_file=tmp.name,
            truncated=True,
            total_lines=total,
        )

    return ExecutionResult(
        command=command,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
        exit_code=exit_code,
        output_file=tmp.name,
        truncated=False,
        total_lines=total,
    )
```

Also add the missing imports at the top of `executor.py` (after `import subprocess`):

```python
import os
import signal
import tempfile
import threading
import time
from typing import Callable
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_executor.py -v
```

Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add src/lama_code/executor.py tests/test_executor.py
git commit -m "feat: executor — execute_streaming() with live stdout/stderr and temp file"
```

---

### Task 3: Add streaming display functions

**Files:**
- Modify: `src/lama_code/display.py`

No new test file — display functions are tested visually (as per original design). Existing tests must still pass.

- [ ] **Step 1: Add four new functions to `src/lama_code/display.py`**

After `end_stream()`, add:

```python
def stream_stdout_line(line: str) -> None:
    sys.stdout.write(line)
    sys.stdout.flush()


def stream_stderr_line(line: str) -> None:
    console.print(f"[yellow]{line.rstrip()}[/yellow]")


def show_stdin_waiting() -> None:
    console.print("[dim]⌨  commande en attente d'entrée...[/dim]")


def show_truncation(lines_shown: int, total_lines: int, filepath: str) -> None:
    console.print(
        f"[dim]... {total_lines - lines_shown} lignes supplémentaires "
        f"dans {filepath}[/dim]"
    )
```

- [ ] **Step 2: Update `show_result()` to handle truncated results**

Replace the existing `show_result()`:

```python
def show_result(result: ExecutionResult) -> None:
    if result.truncated:
        show_truncation(
            lines_shown=result.stdout.count("\n"),
            total_lines=result.total_lines,
            filepath=result.output_file,
        )
        return
    if result.success:
        if result.stdout:
            console.print(f"[green]✓[/green]  {result.stdout.rstrip()}")
    else:
        output = result.stderr.rstrip() or result.stdout.rstrip()
        console.print(f"[red]✗[/red]  [red]{output}[/red]")
```

- [ ] **Step 3: Run full test suite — verify nothing broken**

```bash
python3 -m pytest -v
```

Expected: all existing tests pass (at minimum 29 tests now).

- [ ] **Step 4: Commit**

```bash
git add src/lama_code/display.py
git commit -m "feat: display — streaming output functions and truncation notice"
```

---

### Task 4: Update agent.py — streaming callbacks and _handle_stdin_needed

**Files:**
- Modify: `src/lama_code/agent.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Update fake_execute in test helpers — write failing tests first**

In `tests/test_agent.py`, update `_make_agent()` — the `fake_execute` must now accept keyword args matching the new `execute_streaming` signature:

```python
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

    def fake_execute(cmd, on_stdout=None, on_stderr=None, on_stdin_needed=None):
        return ExecutionResult(command=cmd, stdout=f"output:{cmd}", stderr="", exit_code=0)

    return Agent(cfg=cfg, ollama=ollama, display=display, execute_fn=fake_execute)
```

Also update `test_confirm_skips_block_when_denied`:

```python
def test_confirm_skips_block_when_denied():
    cfg = Config(yolo=False, max_cycles=5, context_window=25)
    ollama = MagicMock()
    display = MagicMock()
    display.confirm.return_value = False

    responses = [
        ["```bash\necho hi\n```"],
        ["La commande a été ignorée."],
    ]
    call_count = [0]
    def generate(messages):
        idx = call_count[0]
        call_count[0] += 1
        return iter(responses[idx] if idx < len(responses) else ["Terminé."])
    ollama.generate.side_effect = generate

    def fake_execute(cmd, on_stdout=None, on_stderr=None, on_stdin_needed=None):
        return ExecutionResult(command=cmd, stdout="output", stderr="", exit_code=0)

    agent = Agent(cfg=cfg, ollama=ollama, display=display, execute_fn=fake_execute)
    agent.run("dis hi")

    display.confirm.assert_called_once()
    assert "[ignoré par l'utilisateur]" in agent.history[2].content
```

Also update `test_build_messages_includes_system_prompt` — its lambda needs the new signature:

```python
    agent = Agent(cfg=cfg, ollama=ollama, display=display,
                  execute_fn=lambda cmd, **kw: ExecutionResult(cmd, "", "", 0))
```

Add new tests at the end:

```python
def test_format_result_truncated():
    from lama_code.agent import Agent
    cfg = Config(yolo=True, max_cycles=5, context_window=25)
    ollama = MagicMock()
    ollama.generate.return_value = iter(["ok"])
    display = MagicMock()
    display.confirm.return_value = True

    truncated_result = ExecutionResult(
        command="seq 1 300",
        stdout="1\n2\n3\n[... 297 lignes supplémentaires dans /tmp/lama-abc.txt\n...]",
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

    # History should contain the truncated output passed to AI
    result_msg = agent.history[2].content
    assert "/tmp/lama-abc.txt" in result_msg


def test_stdin_needed_ai_responds():
    """Verifies _handle_stdin_needed sends output to AI and returns stdin block."""
    cfg = Config(yolo=True, max_cycles=5, context_window=25)
    ollama = MagicMock()
    # First call: bash block. Second call (stdin check): AI sends stdin block.
    # Third call: final response.
    responses = [
        ["```bash\nread -p 'Name: ' n\n```"],
        ["```stdin\nAlice\n```"],
        ["Réponse finale."],
    ]
    call_count = [0]
    def generate(messages):
        idx = call_count[0]
        call_count[0] += 1
        return iter(responses[idx] if idx < len(responses) else ["ok"])
    ollama.generate.side_effect = generate

    display = MagicMock()
    display.confirm.return_value = True
    stdin_injected = []

    def fake_execute(cmd, on_stdout=None, on_stderr=None, on_stdin_needed=None):
        # Simulate the command asking for input
        if on_stdin_needed:
            val = on_stdin_needed("Name: ")
            stdin_injected.append(val)
        return ExecutionResult(command=cmd, stdout="Name: Alice\n", stderr="", exit_code=0)

    agent = Agent(cfg=cfg, ollama=ollama, display=display, execute_fn=fake_execute)
    agent.run("dis mon nom")

    assert stdin_injected[0] == "Alice\n"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python3 -m pytest tests/test_agent.py -v 2>&1 | head -20
```

Expected: failures due to `execute_fn(cmd)` being called with wrong signature.

- [ ] **Step 3: Rewrite `_execute_blocks()` and add `_handle_stdin_needed()` and `_format_result()` in `src/lama_code/agent.py`**

Update the `Agent` class type hint and methods:

```python
class Agent:
    def __init__(
        self,
        cfg: Config,
        ollama,
        display,
        execute_fn: Callable,
    ):
        self.cfg = cfg
        self.ollama = ollama
        self.display = display
        self.execute_fn = execute_fn
        self.history: list[Message] = []

    # ... run(), _build_messages(), _stream() unchanged ...

    def _execute_blocks(self, blocks: list[str]) -> str:
        parts = []
        for cmd in blocks:
            self.display.show_block(cmd)
            if not self.cfg.yolo and not self.display.confirm():
                parts.append(f"$ {cmd}\n[ignoré par l'utilisateur]")
                continue
            result = self.execute_fn(
                cmd,
                on_stdout=self.display.stream_stdout_line,
                on_stderr=self.display.stream_stderr_line,
                on_stdin_needed=self._handle_stdin_needed,
            )
            self.display.show_result(result)
            parts.append(self._format_result(result))
        return "\n\n".join(parts)

    def _handle_stdin_needed(self, output_so_far: str) -> str | None:
        self.display.show_stdin_waiting()
        messages = self._build_messages() + [{
            "role": "user",
            "content": (
                f"[La commande attend une entrée. Output jusqu'ici :]\n{output_so_far}\n\n"
                "Réponds avec ```stdin\\nvaleur\\n``` ou écris 'utilisateur' "
                "pour laisser l'humain répondre."
            ),
        }]
        response = "".join(self.ollama.generate(messages))
        stdin_blocks = re.findall(r"```stdin\n(.*?)```", response, re.DOTALL)
        if stdin_blocks:
            return stdin_blocks[0]
        return None

    def _format_result(self, result) -> str:
        if result.success:
            return f"$ {result.command}\n{result.stdout}"
        return f"$ {result.command}\n[exit {result.exit_code}]\n{result.stderr}"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python3 -m pytest tests/test_agent.py -v
```

Expected: `13 passed` (10 original + 3 new)

- [ ] **Step 5: Run full suite**

```bash
python3 -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/lama_code/agent.py tests/test_agent.py
git commit -m "feat: agent — streaming callbacks, _handle_stdin_needed, _format_result"
```

---

### Task 5: Integration smoke test

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Smoke test — basic live output**

```bash
~/.local/bin/lama-code --yolo "affiche les 5 premiers fichiers de /etc"
```

Expected:
- Spinner tourne pendant réflexion
- `lama▸` apparaît, bloc bash affiché
- Output `/etc` s'affiche ligne par ligne en blanc
- IA résume

- [ ] **Step 3: Smoke test — stderr visible**

```bash
~/.local/bin/lama-code --yolo "ls /nonexistent 2>&1; echo done"
```

Expected: erreur stderr apparaît en jaune, `done` en blanc.

- [ ] **Step 4: Smoke test — output tronqué**

```bash
~/.local/bin/lama-code --yolo "seq 1 500"
```

Expected: les 200 premières lignes s'affichent, puis notice de troncature avec chemin `/tmp/lama-XXXX.txt`.

- [ ] **Step 5: Commit final**

```bash
git add .
git commit -m "feat: live output — streaming, stdin detection, temp file truncation"
git push
```
