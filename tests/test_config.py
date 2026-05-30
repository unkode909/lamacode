from pathlib import Path
from lama_code.config import load_config


def test_defaults_when_no_files(tmp_path):
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project")
    assert cfg.model == "phi4-mini"
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.context_window == 25
    assert cfg.yolo is False
    assert cfg.max_cycles == 10
    assert cfg.system_prompt == ""


def test_global_file_loaded(tmp_path):
    (tmp_path / ".lama.md").write_text(
        "---\nmodel: llama3.2\ncontext_window: 5\n---\nSois concis."
    )
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project")
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
    cfg = load_config(home_dir=home, project_dir=project)
    assert cfg.model == "llama3.2"
    assert cfg.yolo is True
    assert "Global." in cfg.system_prompt
    assert "Project." in cfg.system_prompt


def test_yolo_override(tmp_path):
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path, yolo_override=True)
    assert cfg.yolo is True


def test_model_override(tmp_path):
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path, model_override="mistral")
    assert cfg.model == "mistral"


def test_file_without_frontmatter(tmp_path):
    (tmp_path / ".lama.md").write_text("Juste des instructions.")
    cfg = load_config(home_dir=tmp_path, project_dir=tmp_path / "project")
    assert cfg.system_prompt == "Juste des instructions."
    assert cfg.model == "phi4-mini"
