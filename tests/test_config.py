from pathlib import Path
from unittest.mock import patch
from lama_code.config import load_config, DEFAULT_OLLAMA_URL

# Isolate tests from /etc/lama.md on the actual system
_NO_SYSTEM = Path("/nonexistent/no.md")


def test_defaults_when_no_files(tmp_path):
    with patch("lama_code.config._discover_ollama_url", return_value=DEFAULT_OLLAMA_URL):
        cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project",
                          system_lama=_NO_SYSTEM)
    assert cfg.model == "phi4-mini"
    assert cfg.ollama_url == DEFAULT_OLLAMA_URL
    assert cfg.context_window == 25
    assert cfg.yolo is False
    assert cfg.max_cycles == 10
    assert cfg.system_prompt == ""


def test_global_file_loaded(tmp_path):
    (tmp_path / ".lama.md").write_text(
        "---\nmodel: llama3.2\ncontext_window: 5\n---\nSois concis."
    )
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project",
                      system_lama=_NO_SYSTEM)
    assert cfg.model == "llama3.2"
    assert cfg.context_window == 5
    assert cfg.system_prompt == "Sois concis."


def test_project_overrides_global(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / ".lama.md").write_text("---\nmodel: phi4-mini\nyolo: false\n---\nGlobal.")
    (project / ".lama.md").write_text("---\nmodel: llama3.2\nyolo: true\n---\nProject.")
    cfg = load_config(home_dir=home, project_dir=project, system_lama=_NO_SYSTEM)
    assert cfg.model == "llama3.2"
    assert cfg.yolo is True
    assert "Global." in cfg.system_prompt
    assert "Project." in cfg.system_prompt


def test_yolo_override(tmp_path):
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path,
                      system_lama=_NO_SYSTEM, yolo_override=True)
    assert cfg.yolo is True


def test_model_override(tmp_path):
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path,
                      system_lama=_NO_SYSTEM, model_override="mistral")
    assert cfg.model == "mistral"


def test_file_without_frontmatter(tmp_path):
    (tmp_path / ".lama.md").write_text("Juste des instructions.")
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project",
                      system_lama=_NO_SYSTEM)
    assert cfg.system_prompt == "Juste des instructions."
    assert cfg.model == "phi4-mini"


def test_streaming_defaults(tmp_path):
    with patch("lama_code.config._discover_ollama_url", return_value=DEFAULT_OLLAMA_URL):
        cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project",
                          system_lama=_NO_SYSTEM)
    assert cfg.stdin_timeout == 3.0
    assert cfg.max_output_lines == 200


def test_streaming_params_from_file(tmp_path):
    (tmp_path / ".lama.md").write_text(
        "---\nstdin_timeout: 5.0\nmax_output_lines: 100\n---\n"
    )
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project",
                      system_lama=_NO_SYSTEM)
    assert cfg.stdin_timeout == 5.0
    assert cfg.max_output_lines == 100


def test_system_lama_loaded(tmp_path):
    system = tmp_path / "system.md"
    system.write_text("---\nmodel: system-model\n---\nConfig système.")
    cfg = load_config(home_dir=tmp_path / "home", project_dir=tmp_path / "project",
                      system_lama=system)
    assert cfg.model == "system-model"
    assert "Config système." in cfg.system_prompt
