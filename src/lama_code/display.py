from rich.console import Console
from rich.panel import Panel
from lama_code.executor import ExecutionResult

console = Console()


def print_header(model: str, context_window: int, yolo: bool, lama_md_status: str) -> None:
    yolo_str = "[red]oui[/red]" if yolo else "[green]non[/green]"
    console.print(
        f"[bold]lama-code v0.1.0[/bold]  |  modèle: [cyan]{model}[/cyan]  |  "
        f"yolo: {yolo_str}  |  contexte: [dim]{context_window} échanges[/dim]  |  "
        f"lama.md: [dim]{lama_md_status}[/dim]"
    )
    console.print("─" * 70, style="dim")


def start_stream() -> None:
    console.print("\n[bold cyan]lama▸[/bold cyan] ", end="")


def stream_token(token: str) -> None:
    print(token, end="", flush=True)


def end_stream() -> None:
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
