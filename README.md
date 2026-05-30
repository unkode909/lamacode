# lama-code

Agent IA local propulsé par [Ollama](https://ollama.com). Inspire de Claude Code, mais 100% local et privé.

L'IA peut lire des fichiers, exécuter des commandes bash et interagir avec ton système — sous tes droits, dans ton terminal.

## Fonctionnement

lama-code envoie tes messages à un modèle local via Ollama. Quand l'IA inclut un bloc ` ```bash ``` ` dans sa réponse, lama-code l'exécute (après confirmation) et renvoie le résultat à l'IA, qui continue.

```
vous> liste les fichiers python de ce projet

lama▸ Je vais chercher les fichiers Python.

  ┌─ bash ──────────────────────────────┐
  │ find . -name "*.py" -type f         │
  └─────────────────────────────────────┘
  Exécuter ? [o/N] o

  ✓  ./src/lama_code/agent.py
     ./src/lama_code/cli.py
     ...

Il y a 6 fichiers Python dans ce projet.
```

## Installation

### Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé et en cours d'exécution
- Un modèle Ollama téléchargé (ex: `ollama pull qwen2.5-coder:1.5b`)

### Installer

```bash
git clone https://github.com/unkode909/lamacode
cd lamacode
bash install.sh
```

Le script installe lama-code, configure le PATH et crée `~/.lama.md` si absent.

### Modèles recommandés

| Modèle | Taille | Usage |
|--------|--------|-------|
| `qwen2.5-coder:1.5b` | ~1 GB | Rapide, orienté code — **recommandé** |
| `phi4-mini` | ~2.5 GB | Plus généraliste |
| `qwen2.5-coder:7b` | ~4.5 GB | Puissant, plus lent |

## Utilisation

```bash
lama-code                        # mode REPL interactif
lama-code "refactor ce fichier"  # one-shot
lama-code --yolo "..."           # sans confirmations
lama-code --model phi4-mini      # changer de modèle
lama-code --version
```

## Configuration — `.lama.md`

lama-code charge deux fichiers de config (YAML + instructions pour l'IA) :

- `~/.lama.md` — configuration globale
- `.lama.md` — configuration du projet courant (écrase le global)

```markdown
---
model: qwen2.5-coder:1.5b
context_window: 25
yolo: false
max_cycles: 10
---

Tu es un assistant spécialisé dans ce projet FastAPI.
Utilise toujours des types stricts. Tests avec pytest.
```

### Options de configuration

| Clé | Défaut | Description |
|-----|--------|-------------|
| `model` | `phi4-mini` | Modèle Ollama |
| `ollama_url` | `http://localhost:11434` | URL de l'API Ollama |
| `context_window` | `25` | Échanges gardés en mémoire |
| `yolo` | `false` | Désactive les confirmations |
| `max_cycles` | `10` | Max itérations action→observation par tour |

## Flags CLI

| Flag | Description |
|------|-------------|
| `--yolo` | Désactive les confirmations (dangereux) |
| `--model <nom>` | Remplace le modèle défini dans `.lama.md` |
| `--version` | Affiche la version |
| `--help` | Aide |

## Quitter

En mode REPL : `Ctrl+C` ou `Ctrl+D`
