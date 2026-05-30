from lama_code.executor import execute


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
