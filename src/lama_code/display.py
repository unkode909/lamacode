import sys
import threading
import time
from rich.console import Console
from rich.panel import Panel
from lama_code import __version__
from lama_code.executor import ExecutionResult

console = Console()

_SPINNER_CHARS = ['|', '/', '-', '\\']
_PREFIX = "lama▸ "

class _Spinner:
    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{_SPINNER_CHARS[i % 4]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{_PREFIX}")
        sys.stdout.flush()

_spinner = _Spinner()
_spinner_running = False


def select_model(models: list[str], current: str) -> str:
    """Show a numbered model selection menu. Returns the chosen model name."""
    console.print("\n[bold]Modèles disponibles :[/bold]")
    for i, name in enumerate(models, 1):
        marker = "[cyan]▶[/cyan] " if name == current else "  "
        console.print(f"  {marker}[bold]{i}[/bold]. {name}")
    console.print(f"\n[dim](modèle configuré : {current})[/dim]")
    while True:
        try:
            raw = input("Choisir [1-" + str(len(models)) + "] (Entrée = garder actuel) : ").strip()
        except (KeyboardInterrupt, EOFError):
            return current
        if not raw:
            return current
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        console.print("[red]Choix invalide.[/red]")


def select_yolo(current: bool) -> bool:
    """Ask user to choose yolo mode. Interactive Enter defaults to yolo on."""
    console.print("\n[bold]Mode d'exécution :[/bold]")
    options = [("Yolo (sans confirmation)", True), ("Confirmer chaque commande", False)]
    for i, (label, val) in enumerate(options, 1):
        marker = "[cyan]▶[/cyan] " if val == current else "  "
        console.print(f"  {marker}[bold]{i}[/bold]. {label}")
    try:
        raw = input("Choisir [1-2] (Entrée = yolo) : ").strip()
    except (KeyboardInterrupt, EOFError):
        # Non-interactive context — keep the configured value, don't force yolo
        return current
    if raw == "2":
        return False
    return True  # "1", empty, or anything else → yolo


def print_header(model: str, context_window: int, yolo: bool, lama_md_status: str) -> None:
    yolo_str = "[red]oui[/red]" if yolo else "[green]non[/green]"
    console.print(
        f"[bold]lama-code v{__version__}[/bold]  |  modèle: [cyan]{model}[/cyan]  |  "
        f"yolo: {yolo_str}  |  contexte: [dim]{context_window} échanges[/dim]  |  "
        f"lama.md: [dim]{lama_md_status}[/dim]"
    )
    console.print("─" * 70, style="dim")


def start_stream() -> None:
    global _spinner_running
    _spinner_running = True
    _spinner.start()


def stream_token(token: str) -> None:
    global _spinner_running
    if _spinner_running:
        _spinner.stop()
        _spinner_running = False
    print(token, end="", flush=True)


def end_stream() -> None:
    global _spinner_running
    if _spinner_running:
        _spinner.stop()
        _spinner_running = False
    print()


def show_command_running(command: str) -> None:
    console.print(f"[green]$ {command}[/green]")


def stream_stdout_line(line: str) -> None:
    sys.stdout.write(line)
    sys.stdout.flush()


def stream_stderr_line(line: str) -> None:
    console.print(f"[yellow]{line.rstrip()}[/yellow]")


def show_stdin_waiting() -> None:
    console.print("[dim]⌨  commande en attente d'entrée...[/dim]")


def show_stdin_proposed(value: str) -> None:
    console.print("[dim]L'IA propose d'envoyer :[/dim]")
    console.print(Panel(value.rstrip(), title="[cyan]stdin proposé[/cyan]", border_style="yellow", expand=False))


def confirm_stdin() -> bool:
    try:
        answer = input("  Envoyer ? [o/N] ").strip().lower()
        return answer in ("o", "oui", "y", "yes")
    except EOFError:
        return False


def show_truncation(lines_shown: int, total_lines: int, filepath: str) -> None:
    console.print(
        f"[dim]... {total_lines - lines_shown} lignes supplémentaires "
        f"dans {filepath}[/dim]"
    )


def show_block(command: str) -> None:
    console.print(
        Panel(command, title="[cyan]bash[/cyan]", border_style="cyan", expand=False)
    )


def confirm() -> bool:
    try:
        answer = input("  Exécuter ? [o/N] ").strip().lower()
        return answer in ("o", "oui", "y", "yes")
    except EOFError:
        return False


def show_result(result: ExecutionResult) -> None:
    if result.truncated:
        show_truncation(
            lines_shown=result.stdout.count("\n"),
            total_lines=result.total_lines,
            filepath=result.output_file,
        )
        return
    if result.success:
        if result.stdout:
            console.print(f"[green]✓[/green]  {result.stdout.rstrip()}")
    else:
        output = result.stderr.rstrip() or result.stdout.rstrip()
        console.print(f"[red]✗[/red]  [red]{output}[/red]")


def print_error(message: str) -> None:
    console.print(f"[bold red]Erreur :[/bold red] {message}")
