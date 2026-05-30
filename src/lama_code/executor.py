import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable


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


def execute(command: str) -> ExecutionResult:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        return ExecutionResult(
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            command=command,
            stdout="",
            stderr="[timeout: 60s dépassé]",
            exit_code=124,
        )


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
                    stdin_requested = False
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
            f"[... {total - max_output_lines} lignes supplémentaires — voir {tmp.name}]"
        )
        return ExecutionResult(
            command=command,
            stdout=kept + truncated_msg,
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
