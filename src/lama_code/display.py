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
    if result.success:
        if result.stdout:
            console.print(f"[green]✓[/green]  {result.stdout.rstrip()}")
    else:
        output = result.stderr.rstrip() or result.stdout.rstrip()
        console.print(f"[red]✗[/red]  [red]{output}[/red]")


def print_error(message: str) -> None:
    console.print(f"[bold red]Erreur :[/bold red] {message}")
