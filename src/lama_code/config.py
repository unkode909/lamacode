from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    model: str = "phi4-mini"
    ollama_url: str = "http://localhost:11434"
    context_window: int = 25
    yolo: bool = False
    max_cycles: int = 10
    system_prompt: str = ""


def load_config(
    home_dir: Path = None,
    project_dir: Path = None,
    yolo_override: bool = False,
    model_override: str = None,
) -> Config:
    home_dir = home_dir or Path.home()
    project_dir = project_dir or Path.cwd()

    merged: dict = {}
    bodies: list[str] = []

    for path in [home_dir / ".lama.md", project_dir / ".lama.md"]:
        if path.exists():
            front, body = _parse(path)
            merged.update(front)
            if body:
                bodies.append(body)

    cfg = Config(
        model=merged.get("model", "phi4-mini"),
        ollama_url=merged.get("ollama_url", "http://localhost:11434"),
        context_window=int(merged.get("context_window", 25)),
        yolo=bool(merged.get("yolo", False)),
        max_cycles=int(merged.get("max_cycles", 10)),
        system_prompt="\n\n".join(bodies),
    )

    if yolo_override:
        cfg.yolo = True
    if model_override:
        cfg.model = model_override

    return cfg


def _parse(path: Path) -> tuple[dict, str]:
    content = path.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}, parts[2].strip()
    return {}, content.strip()
