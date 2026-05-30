import argparse
import sys
from pathlib import Path

from lama_code import __version__
from lama_code.config import load_config
from lama_code.ollama import OllamaClient, OllamaError
from lama_code.executor import execute
from lama_code.agent import Agent
from lama_code import display as _display


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lama-code",
        description="Agent IA local propulsé par Ollama",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt one-shot (sans arg = REPL)")
    parser.add_argument("--yolo", action="store_true", help="Désactive les confirmations")
    parser.add_argument("--model", help="Modèle Ollama à utiliser")
    parser.add_argument("--version", action="version", version=f"lama-code {__version__}")
    return parser.parse_args(argv)


def _lama_md_status() -> str:
    has_global = (Path.home() / ".lama.md").exists()
    has_project = (Path.cwd() / ".lama.md").exists()
    if has_global and has_project:
        return "global + projet"
    if has_global:
        return "global"
    if has_project:
        return "projet"
    return "aucun"


def main() -> None:
    args = parse_args()
    cfg = load_config(yolo_override=args.yolo, model_override=args.model)
    ollama = OllamaClient(base_url=cfg.ollama_url, model=cfg.model)
    agent = Agent(cfg=cfg, ollama=ollama, display=_display, execute_fn=execute)

    if args.prompt:
        try:
            agent.run(args.prompt)
        except OllamaError as e:
            _display.print_error(str(e))
            sys.exit(1)
        return

    _display.print_header(
        model=cfg.model,
        context_window=cfg.context_window,
        yolo=cfg.yolo,
        lama_md_status=_lama_md_status(),
    )

    while True:
        try:
            user_input = input("\nvous> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user_input:
            continue
        try:
            agent.run(user_input)
        except KeyboardInterrupt:
            print()
            break
        except OllamaError as e:
            _display.print_error(str(e))
