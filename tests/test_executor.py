import os
from lama_code.executor import execute, execute_streaming


def test_successful_command():
    result = execute("echo hello")
    assert result.success is True
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""
    assert result.exit_code == 0


def test_failing_command():
    result = execute("ls /nonexistent_xyz_path")
    assert result.success is False
    assert result.exit_code != 0
    assert result.stderr != ""


def test_stdout_captured():
    result = execute("printf 'line1\nline2\n'")
    assert "line1" in result.stdout
    assert "line2" in result.stdout


def test_command_stored():
    result = execute("echo test")
    assert result.command == "echo test"


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


def test_streaming_output_file_only_when_truncated():
    # No file for small output
    result = execute_streaming(
        "echo hello",
        on_stdout=lambda l: None,
        on_stderr=lambda l: None,
        on_stdin_needed=lambda _: None,
    )
    assert result.output_file == ""
    assert result.truncated is False

    # File created only when output exceeds max_output_lines
    result2 = execute_streaming(
        "seq 1 300",
        on_stdout=lambda l: None,
        on_stderr=lambda l: None,
        on_stdin_needed=lambda _: None,
        max_output_lines=50,
    )
    assert result2.output_file != ""
    assert os.path.exists(result2.output_file)
    assert result2.truncated is True


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
