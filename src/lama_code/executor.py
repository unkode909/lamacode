import subprocess
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int

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
