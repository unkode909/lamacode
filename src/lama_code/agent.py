import re
from dataclasses import dataclass
from typing import Callable
from lama_code.config import Config
from lama_code.executor import ExecutionResult

BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)```", re.DOTALL)

TOOL_INSTRUCTIONS = """Pour interagir avec le système (fichiers, commandes), utilise des blocs bash :

```bash
commande ici
```

Le résultat te sera renvoyé automatiquement. Tu peux enchaîner plusieurs blocs par réponse.
Pour des réponses purement textuelles, écris sans blocs bash."""


@dataclass
class Message:
    role: str
    content: str


def parse_bash_blocks(text: str) -> list[str]:
    return [m.group(1).strip() for m in BASH_BLOCK_RE.finditer(text)]


class Agent:
    def __init__(
        self,
        cfg: Config,
        ollama,
        display,
        execute_fn: Callable[[str], ExecutionResult],
    ):
        self.cfg = cfg
        self.ollama = ollama
        self.display = display
        self.execute_fn = execute_fn
        self.history: list[Message] = []

    def run(self, user_input: str) -> str:
        self.history.append(Message("user", user_input))

        for _ in range(self.cfg.max_cycles):
            messages = self._build_messages()
            response = self._stream(messages)
            blocks = parse_bash_blocks(response)
            self.history.append(Message("assistant", response))

            if not blocks:
                return response

            results = self._execute_blocks(blocks)
            self.history.append(Message("user", f"[Résultats]\n{results}"))

        return "[Max cycles atteint]"

    def _build_messages(self) -> list[dict]:
        system = TOOL_INSTRUCTIONS
        if self.cfg.system_prompt:
            system = self.cfg.system_prompt + "\n\n" + TOOL_INSTRUCTIONS
        messages = [{"role": "system", "content": system}]
        window = self.history[-(self.cfg.context_window * 2):]
        messages += [{"role": m.role, "content": m.content} for m in window]
        return messages

    def _stream(self, messages: list[dict]) -> str:
        full = ""
        self.display.start_stream()
        for token in self.ollama.generate(messages):
            self.display.stream_token(token)
            full += token
        self.display.end_stream()
        return full

    def _execute_blocks(self, blocks: list[str]) -> str:
        parts = []
        for cmd in blocks:
            self.display.show_block(cmd)
            if not self.cfg.yolo and not self.display.confirm():
                parts.append(f"$ {cmd}\n[ignoré par l'utilisateur]")
                continue
            result = self.execute_fn(cmd)
            self.display.show_result(result)
            if result.success:
                parts.append(f"$ {cmd}\n{result.stdout}")
            else:
                parts.append(f"$ {cmd}\n[exit {result.exit_code}]\n{result.stderr}")
        return "\n\n".join(parts)
