#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== lama-code installer ==="
echo

# Python 3.10+
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "Erreur : Python 3.10+ requis." >&2
    exit 1
fi

PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
PY_VERSION="$PY_MAJOR.$PY_MINOR"

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

# ── Installation ─────────────────────────────────────────────────────────────

if [ "$(id -u)" -eq 0 ]; then
    echo "Installation globale (root)..."

    # Installe le package en mode éditable dans le site-packages système
    $PYTHON -m pip install -e "$REPO_DIR" --break-system-packages -q

    # Crée un wrapper dans /usr/local/bin (accessible à tous les users)
    WRAPPER=/usr/local/bin/lama-code
    cat > "$WRAPPER" <<WEOF
#!/usr/bin/env bash
exec "$PYTHON" -m lama_code "\$@"
WEOF
    chmod 755 "$WRAPPER"
    echo "✓ lama-code installé dans /usr/local/bin (tous les utilisateurs)"

    # Crée /etc/lama.md — config système partagée (lue avant ~/.lama.md)
    SYSTEM_LAMA=/etc/lama.md
    if [ ! -f "$SYSTEM_LAMA" ]; then
        cat > "$SYSTEM_LAMA" <<'EOF'
---
model: qwen2.5-coder:7b
ollama_url: http://localhost:11434
context_window: 25
yolo: true
max_cycles: 10
stdin_timeout: 30
---

You are lama-code, a local execution agent on Linux.
You EXECUTE commands — you do not explain, you do not announce, you act.
Short responses. Zero fluff. If it can be done with a command, run it.
EOF
        chmod 644 "$SYSTEM_LAMA"
        echo "✓ /etc/lama.md créé (config système partagée)"
    else
        echo "✓ /etc/lama.md déjà présent"
    fi

else
    echo "Installation locale (user: $USER)..."
    $PYTHON -m pip install -e "$REPO_DIR" --break-system-packages -q
    echo "✓ lama-code installé"

    # Ajoute ~/.local/bin au PATH si nécessaire
    LOCAL_BIN="$HOME/.local/bin"
    if ! echo "$PATH" | grep -q "$LOCAL_BIN"; then
        SHELL_RC="$HOME/.bashrc"
        [ -n "$ZSH_VERSION" ] && SHELL_RC="$HOME/.zshrc"
        if ! grep -q '\.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            echo "✓ PATH mis à jour dans $SHELL_RC (relance: source $SHELL_RC)"
        fi
    else
        echo "✓ PATH déjà configuré"
    fi

    # ~/.lama.md personnel
    LAMA_MD="$HOME/.lama.md"
    if [ ! -f "$LAMA_MD" ]; then
        cat > "$LAMA_MD" <<'EOF'
---
model: qwen2.5-coder:7b
ollama_url: http://localhost:11434
context_window: 25
yolo: true
max_cycles: 10
stdin_timeout: 30
---

You are lama-code, a local execution agent on Linux.
You EXECUTE commands — you do not explain, you do not announce, you act.
Short responses. Zero fluff. If it can be done with a command, run it.
EOF
        echo "✓ $LAMA_MD créé"
    else
        echo "✓ $LAMA_MD déjà présent"
    fi
fi

echo
echo "=== Installation terminée ==="
echo
echo "Lance lama-code :"
echo "  lama-code                       # mode REPL interactif"
echo "  lama-code \"liste les fichiers\"   # one-shot"
echo "  lama-code --yolo \"...\"          # sans confirmations"
echo "  lama-code --model phi4-mini     # changer de modèle"
echo
echo "Modèles Ollama recommandés :"
echo "  ollama pull qwen2.5-coder:1.5b  # rapide, orienté code (défaut)"
echo "  ollama pull phi4-mini            # plus généraliste"
