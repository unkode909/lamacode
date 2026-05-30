import pytest
from lama_code.cli import parse_args


def test_repl_mode_no_args():
    args = parse_args([])
    assert args.prompt is None
    assert args.yolo is False
    assert args.model is None


def test_oneshot_mode():
    args = parse_args(["dis bonjour"])
    assert args.prompt == "dis bonjour"


def test_yolo_flag():
    args = parse_args(["--yolo"])
    assert args.yolo is True


def test_model_flag():
    args = parse_args(["--model", "llama3.2"])
    assert args.model == "llama3.2"


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
