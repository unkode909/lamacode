# lama-code — Live Output Design Spec

**Date:** 2026-05-30  
**Statut:** Approuvé

---

## Vue d'ensemble

Remplace l'exécution bloquante (`subprocess.run`) par un mode streaming ligne par ligne via `Popen` + threads. L'output est affiché en live, stdout en blanc et stderr en jaune. L'output complet est sauvegardé dans un fichier temporaire `/tmp/lama-XXXXX.txt`. L'IA reçoit une version tronquée avec référence au fichier. La commande peut demander une entrée utilisateur : l'IA est consultée et peut injecter une réponse, ou l'utilisateur tape manuellement. Ctrl+C interrompt proprement.

---

## 1. Architecture & fichiers modifiés

```
src/lama_code/
├── executor.py   → execute_streaming() avec Popen + threads + détection stdin
├── display.py    → stream_stdout_line(), stream_stderr_line(), show_truncation()
└── agent.py      → _execute_blocks() appelle execute_streaming() avec callbacks
```

**Flux général :**
```
agent demande exécution
  → execute_streaming(command, on_stdout, on_stderr, on_stdin_needed)
      stdout thread → on_stdout(line) → display blanc + buffer + fichier tmp
      stderr thread → on_stderr(line) → display jaune + buffer + fichier tmp
      [silence > stdin_timeout sec + process vivant]
        → on_stdin_needed(output_so_far)
            → agent envoie à l'IA : "la commande attend une entrée"
            → IA répond avec ```stdin\nvaleur\n``` OU "utilisateur"
            → stdin injecté au processus OU input() appelé
  → retourne ExecutionResult(stdout, stderr, output_file, truncated, total_lines)
```

**Pas de dépendance circulaire** : `executor.py` ne connaît pas `agent.py` — il reçoit des callbacks. `agent.py` fournit les callbacks et reste le coordinateur.

`execute()` (non-streaming) reste intact pour les tests existants.

---

## 2. `executor.py` : execute_streaming()

### Signature

```python
def execute_streaming(
    command: str,
    on_stdout: Callable[[str], None],
    on_stderr: Callable[[str], None],
    on_stdin_needed: Callable[[str], str | None],
    stdin_timeout: float = 3.0,
    max_output_lines: int = 200,
) -> ExecutionResult
```

### ExecutionResult — nouveaux champs

```python
@dataclass
class ExecutionResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    output_file: str = ""      # chemin /tmp/lama-XXXXX.txt
    truncated: bool = False    # output tronqué pour l'IA
    total_lines: int = 0       # total lignes capturées
```

### Logique interne

1. `Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)`
2. Ouvre `/tmp/lama-XXXXX.txt` via `tempfile.NamedTemporaryFile(delete=False)`
3. Thread stdout : lit ligne par ligne → `on_stdout(line)` + écrit fichier tmp + buffer
4. Thread stderr : idem → `on_stderr(line)` + fichier tmp + buffer
5. Thread principal : si `time.time() - last_line_time > stdin_timeout` et process vivant → `on_stdin_needed(buffer_so_far)` → reçoit `str` (injecté dans stdin) ou `None` (user tape via `input()`)
6. Ctrl+C : `process.send_signal(SIGINT)`, threads stoppés proprement
7. À la fin : si `total_lines > max_output_lines` → `truncated=True`

---

## 3. `display.py` : nouvelles fonctions

```python
def stream_stdout_line(line: str) -> None:
    # blanc — output standard
    sys.stdout.write(line)
    sys.stdout.flush()

def stream_stderr_line(line: str) -> None:
    # jaune — erreurs/warnings
    console.print(f"[yellow]{line.rstrip()}[/yellow]")

def show_stdin_waiting() -> None:
    console.print("[dim]⌨  commande en attente d'entrée...[/dim]")

def show_truncation(lines_shown: int, total_lines: int, filepath: str) -> None:
    console.print(
        f"[dim]... {total_lines - lines_shown} lignes supplémentaires "
        f"dans {filepath}[/dim]"
    )
```

**`show_result()` modifiée :** si `result.truncated`, affiche `show_truncation()` au lieu du contenu complet. Comportement actuel inchangé si non tronqué.

---

## 4. `agent.py` : callbacks et bloc stdin

### `_execute_blocks()` modifiée

```python
def _execute_blocks(self, blocks: list[str]) -> str:
    parts = []
    for cmd in blocks:
        self.display.show_block(cmd)
        if not self.cfg.yolo and not self.display.confirm():
            parts.append(f"$ {cmd}\n[ignoré par l'utilisateur]")
            continue
        result = self.execute_fn(
            cmd,
            on_stdout=self.display.stream_stdout_line,
            on_stderr=self.display.stream_stderr_line,
            on_stdin_needed=self._handle_stdin_needed,
        )
        self.display.show_result(result)
        parts.append(self._format_result(result))
    return "\n\n".join(parts)
```

### `_handle_stdin_needed(output_so_far: str) -> str | None`

Appelé quand la commande attend une entrée :

```python
def _handle_stdin_needed(self, output_so_far: str) -> str | None:
    self.display.show_stdin_waiting()
    messages = self._build_messages() + [{
        "role": "user",
        "content": (
            f"[La commande attend une entrée. Output jusqu'ici :]\n{output_so_far}\n\n"
            "Réponds avec ```stdin\\nvaleur\\n``` ou écris 'utilisateur' "
            "pour laisser l'humain répondre."
        )
    }]
    response = "".join(self.ollama.generate(messages))
    stdin_blocks = re.findall(r"```stdin\n(.*?)```", response, re.DOTALL)
    if stdin_blocks:
        return stdin_blocks[0]
    return None  # l'utilisateur tape manuellement
```

### `_format_result()` — texte envoyé à l'IA

- Non tronqué : stdout + stderr complets
- Tronqué : premières N lignes + message :
  ```
  [... X lignes dans /tmp/lama-XXXXX.txt
  Lis des sections avec : head -n 50 /tmp/lama-XXXXX.txt
                          sed -n '200,250p' /tmp/lama-XXXXX.txt]
  ```

### Regex mis à jour

`BASH_BLOCK_RE` ignore les blocs ` ```stdin ` — ils ne sont pas exécutés.

---

## 5. Configuration `.lama.md`

| Clé | Défaut | Description |
|-----|--------|-------------|
| `stdin_timeout` | `3.0` | Secondes de silence avant détection d'attente stdin |
| `max_output_lines` | `200` | Lignes max envoyées à l'IA avant troncature |

---

## 6. Hors scope

- PTY / support ANSI complet des programmes interactifs (ncurses, etc.)
- Streaming en temps réel vers l'IA (ligne par ligne)
- Historique des fichiers tmp entre sessions
