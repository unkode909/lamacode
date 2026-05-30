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

# Installation globale (requiert root) ou locale (user courant)
if [ "$(id -u)" -eq 0 ]; then
    echo "Installation globale (root détecté)..."
    $PYTHON -m pip install -e "$REPO_DIR" --break-system-packages -q

    # Wrapper dans /usr/local/bin accessible à tous
    WRAPPER=/usr/local/bin/lama-code
    LAMA_BIN=$($PYTHON -c "import shutil; print(shutil.which('lama-code') or '')" 2>/dev/null)

    # Si pip n'a pas mis le bin dans /usr/local/bin, créer un wrapper
    if [ "$LAMA_BIN" != "$WRAPPER" ]; then
        cat > "$WRAPPER" <<WEOF
#!/usr/bin/env bash
exec $PYTHON -m lama_code "\$@"
WEOF
        chmod 755 "$WRAPPER"
    fi

    echo "✓ lama-code installé dans /usr/local/bin (accessible à tous les utilisateurs)"

    # ~/.lama.md pour root
    LAMA_MD="$HOME/.lama.md"
else
    echo "Installation locale (user: $USER)..."
    $PYTHON -m pip install -e "$REPO_DIR" --break-system-packages -q
    echo "✓ lama-code installé"

    # PATH
    LOCAL_BIN="$HOME/.local/bin"
    if ! echo "$PATH" | grep -q "$LOCAL_BIN"; then
        SHELL_RC="$HOME/.bashrc"
        [ -n "$ZSH_VERSION" ] && SHELL_RC="$HOME/.zshrc"
        if ! grep -q '\.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            echo "✓ PATH mis à jour dans $SHELL_RC (relance ton shell ou: source $SHELL_RC)"
        fi
    else
        echo "✓ PATH déjà configuré"
    fi

    LAMA_MD="$HOME/.lama.md"
fi

# ~/.lama.md pour l'utilisateur courant
if [ ! -f "$LAMA_MD" ]; then
    cat > "$LAMA_MD" <<'EOF'
---
model: qwen2.5-coder:1.5b
context_window: 25
yolo: false
max_cycles: 10
---

Tu es lama-code, un agent d'exécution local sur Linux.
Tu EXÉCUTES des commandes — tu n'expliques pas, tu n'annonces pas, tu agis.
Réponses courtes. Zéro blabla. Si c'est faisable avec une commande, lance-la.
EOF
    echo "✓ $LAMA_MD créé (modèle: qwen2.5-coder:1.5b)"
else
    echo "✓ $LAMA_MD déjà présent"
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
