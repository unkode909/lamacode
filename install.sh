#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMA_MD="$HOME/.lama.md"

echo "=== lama-code installer ==="
echo

# Python 3.10+
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "Erreur : Python 3.10+ requis." >&2
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "Erreur : Python 3.10+ requis (trouvé $PY_VERSION)." >&2
    exit 1
fi

echo "✓ Python $PY_VERSION"

# Ollama
if ! command -v ollama &>/dev/null; then
    echo "⚠ Ollama non trouvé. Installe-le depuis https://ollama.com"
else
    echo "✓ Ollama détecté"
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "✓ Ollama en cours d'exécution"
    else
        echo "⚠ Ollama installé mais non démarré (ollama serve)"
    fi
fi

echo

# Install package
echo "Installation de lama-code..."
$PYTHON -m pip install -e "$REPO_DIR" --break-system-packages -q
echo "✓ lama-code installé"

# PATH
LOCAL_BIN="$HOME/.local/bin"
if ! echo "$PATH" | grep -q "$LOCAL_BIN"; then
    SHELL_RC="$HOME/.bashrc"
    [ -n "$ZSH_VERSION" ] && SHELL_RC="$HOME/.zshrc"
    if ! grep -q 'LOCAL_BIN\|\.local/bin' "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo "✓ PATH mis à jour dans $SHELL_RC (relance ton shell ou: source $SHELL_RC)"
    fi
else
    echo "✓ PATH déjà configuré"
fi

# ~/.lama.md global
if [ ! -f "$LAMA_MD" ]; then
    cat > "$LAMA_MD" <<'EOF'
---
model: qwen2.5-coder:1.5b
context_window: 25
yolo: false
max_cycles: 10
---

Tu es lama-code, un assistant de développement local sur Linux.
Tu travailles dans le dossier courant, sous les droits de l'utilisateur.
Sois concis. Préfère les petites étapes vérifiables.
EOF
    echo "✓ ~/.lama.md créé (modèle: qwen2.5-coder:1.5b)"
else
    echo "✓ ~/.lama.md déjà présent"
fi

echo
echo "=== Installation terminée ==="
echo
echo "Lance lama-code :"
echo "  lama-code                    # mode REPL interactif"
echo "  lama-code \"liste les fichiers\"  # one-shot"
echo "  lama-code --yolo \"...\"       # sans confirmations"
echo "  lama-code --model phi4-mini  # changer de modèle"
echo
echo "Modèles Ollama recommandés :"
echo "  ollama pull qwen2.5-coder:1.5b  # rapide, orienté code (défaut)"
echo "  ollama pull phi4-mini            # plus généraliste"
