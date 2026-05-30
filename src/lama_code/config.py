import dataclasses
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
import yaml

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _discover_ollama_url() -> str:
    # 1. Variable d'environnement standard d'Ollama
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if host:
        if not host.startswith("http"):
            host = f"http://{host}"
        return host.rstrip("/")

    # 2. Sonde HTTP — essaie les ports candidates dans l'ordre
    import urllib.request as _req

    candidates = [DEFAULT_OLLAMA_URL]

    # Cherche d'autres ports via ss
    try:
        out = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=2
        ).stdout
        for line in out.splitlines():
            if "ollama" not in line:
                continue
            for part in line.split():
                if ":" in part:
                    port = part.rsplit(":", 1)[-1]
                    if port.isdigit():
                        url = f"http://localhost:{port}"
                        if url not in candidates:
                            candidates.append(url)
    except Exception:
        pass

    for url in candidates:
        try:
            _req.urlopen(f"{url}/api/tags", timeout=2).close()
            return url
        except Exception:
            continue

    return DEFAULT_OLLAMA_URL


@dataclass
class Config:
    model: str = "phi4-mini"
    ollama_url: str = DEFAULT_OLLAMA_URL
    context_window: int = 25
    yolo: bool = False
    max_cycles: int = 10
    system_prompt: str = ""
    stdin_timeout: float = 3.0
    max_output_lines: int = 200


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

    seen: set = set()
    for path in [Path("/etc/lama.md"), home_dir / ".lama.md", project_dir / ".lama.md"]:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            front, body = _parse(path)
            merged.update(front)
            if body:
                bodies.append(body)

    # ollama_url: fichier .lama.md > OLLAMA_HOST env > auto-découverte > défaut
    if "ollama_url" not in merged:
        merged["ollama_url"] = _discover_ollama_url()

    overrides = {
        k: merged[k]
        for k in ("model", "ollama_url", "yolo")
        if k in merged
    }
    if "context_window" in merged:
        overrides["context_window"] = int(merged["context_window"])
    if "max_cycles" in merged:
        overrides["max_cycles"] = int(merged["max_cycles"])
    if "stdin_timeout" in merged:
        overrides["stdin_timeout"] = float(merged["stdin_timeout"])
    if "max_output_lines" in merged:
        overrides["max_output_lines"] = int(merged["max_output_lines"])
    overrides["system_prompt"] = "\n\n".join(bodies)
    cfg = dataclasses.replace(Config(), **overrides)

    if yolo_override:
        cfg.yolo = True
    if model_override:
        cfg.model = model_override

    return cfg


def _parse(path: Path) -> tuple[dict, str]:
    content = path.read_text(encoding="utf-8")
    if content.startswith("---\n"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}, parts[2].strip()
    return {}, content.strip()
