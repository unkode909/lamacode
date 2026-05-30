# lama-code — Design Spec

**Date:** 2026-05-30  
**Statut:** Approuvé

---

## Vue d'ensemble

`lama-code` est un agent IA local propulsé par Ollama, permettant à un modèle de langage local d'interagir avec le système de fichiers et le terminal, sous les droits de l'utilisateur courant. Inspiré de Claude Code, mais entièrement local.

---

## 1. Architecture & modules

```
lamacode/
├── pyproject.toml
└── src/lama_code/
    ├── __init__.py
    ├── __main__.py       # python -m lama_code
    ├── cli.py            # parsing args, dispatch REPL vs one-shot
    ├── config.py         # chargement et fusion des lama.md
    ├── ollama.py         # client HTTP Ollama (streaming)
    ├── agent.py          # boucle ReAct principale
    ├── executor.py       # exécution bash, confirmations
    └── display.py        # affichage terminal (rich)
```

**Installation :** `pip install -e /home/user/lamacode` → commande `lama-code` disponible globalement.

**Entry point :** `lama-code` → `cli.py:main()`

**Modes :**
- Sans argument → REPL interactif
- Avec argument → one-shot (`lama-code "refactor ce fichier"`)

**Flags CLI :**
- `--model <nom>` — override le modèle Ollama
- `--yolo` — désactive les confirmations
- `--version` — affiche la version

---

## 2. Boucle agent (ReAct loop)

```
Input utilisateur
      │
      ▼
Construire messages (system prompt + historique + input)
      │
      ▼
Ollama API (streaming)
      │
      ▼
Parser la réponse
      │
   ┌──┴──────────────────┐
blocs ```bash```       texte pur
      │                    │
      ▼                    ▼
Pour chaque bloc :      Afficher réponse finale
  • Afficher bloc        Attendre prochain input
  • Demander OK? (sauf --yolo)
  • Exécuter
  • Capturer stdout/stderr
      │
      ▼
Ajouter résultats dans l'historique → retour au début
```

**Règles de la boucle :**

- **Streaming** : réponse affichée token par token, parsing à la fin
- **Multi-bloc** : plusieurs blocs bash dans une réponse sont exécutés en séquence
- **Erreurs** : si exit code ≠ 0, le stderr est quand même renvoyé à l'IA — elle décide quoi faire
- **Context window** : historique tronqué aux N derniers échanges complets (user+assistant). Défaut : 25
- **Anti-boucle infinie** : max `max_cycles` (défaut 10) itérations action→observation par tour

---

## 3. Configuration & lama.md

Les fichiers `lama.md` ont deux parties : un **frontmatter YAML** (config machine) suivi d'un **corps Markdown** (instructions pour l'IA, injectées dans le system prompt).

### Emplacements

| Fichier | Portée |
|---------|--------|
| `~/.lama.md` | Global — toujours chargé |
| `.lama.md` | Projet — chargé si présent dans le dossier courant |

### Exemple — global (`~/.lama.md`)

```markdown
---
model: phi4-mini
ollama_url: http://localhost:11434
context_window: 25
yolo: false
max_cycles: 10
---

Tu es lama-code, un assistant de développement local.
Tu travailles sous les droits de l'utilisateur courant.
Sois concis. Préfère les petites actions vérifiables.
```

### Exemple — projet (`.lama.md`)

```markdown
---
model: llama3.2
yolo: true
---

Ce projet est une API FastAPI.
Utilise toujours des types stricts. Tests avec pytest.
```

### Règles de fusion

- Les clés YAML du projet **écrasent** le global
- Les corps Markdown sont **concaténés** (global d'abord, projet ensuite)
- Le corps combiné devient le **system prompt** envoyé à Ollama

### Paramètres disponibles

| Clé | Défaut | Description |
|-----|--------|-------------|
| `model` | `phi4-mini` | Modèle Ollama |
| `ollama_url` | `http://localhost:11434` | URL de l'API |
| `context_window` | `25` | Nb d'échanges gardés en mémoire |
| `yolo` | `false` | Désactive les confirmations |
| `max_cycles` | `10` | Max itérations action→observation par tour |

---

## 4. Affichage & UX

### En-tête REPL

```
lama-code v0.1.0  |  modèle: phi4-mini  |  dossier: ~/monprojet
lama.md: global + projet  |  yolo: non  |  contexte: 25 échanges
──────────────────────────────────────────────────────────────────
vous> 
```

### Interaction type

```
vous> liste les fichiers python de ce dossier

lama▸ Je vais lister les fichiers Python présents.

  ┌─ bash ──────────────────────────────┐
  │ find . -name "*.py" -type f         │
  └─────────────────────────────────────┘
  Exécuter ? [o/N] o

  ✓  ./src/lama_code/agent.py
     ./src/lama_code/cli.py
     ./tests/test_agent.py

Il y a 3 fichiers Python dans ce projet.
```

### Codes couleur (via `rich`)

| Élément | Couleur |
|---------|---------|
| Blocs bash | Fond sombre, bordure cyan |
| `✓` résultat OK | Vert |
| `✗` erreur (stderr) | Rouge |
| Réponse IA | Blanc |
| Méta-info (modèle, yolo…) | Gris |

### Mode one-shot

```bash
$ lama-code "combien de lignes dans agent.py"
# Même affichage, quitte après la réponse finale
```

**Dépendance :** `rich` uniquement pour l'affichage.

---

## 5. Dépendances

```toml
[project]
dependencies = [
    "rich>=13.0",
]
```

Pas de dépendance externe au-delà de `rich` — tout le reste est stdlib (urllib, json, subprocess, threading).

---

## 6. Gestion d'erreurs

- **Ollama inaccessible** : message d'erreur explicite (`Erreur : impossible de joindre Ollama sur http://localhost:11434`) et sortie avec code 1
- **Modèle introuvable** : Ollama retourne une erreur — affichée telle quelle, sortie avec code 1
- **Commande bash échoue** : stderr affiché en rouge, renvoyé à l'IA, la boucle continue
- **Ctrl+C en REPL** : interruption propre, pas de crash

---

## 7. Hors scope (v0.1)

- Authentification ou chiffrement
- Plugins ou outils personnalisés
- Support multi-sessions simultanées
- Interface web
- Support de modèles non-Ollama
