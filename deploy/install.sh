#!/usr/bin/env bash
# First-time setup on a fresh VPS.
# Usage: bash deploy/install.sh
set -euo pipefail

REPO_DIR="$HOME/eidolon-telegram"

echo "=== Eidolon Install ==="

# Ensure Python 3.12+
python3 --version

# Create venv if missing
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$REPO_DIR/.venv"
fi

echo "Installing dependencies..."
"$REPO_DIR/.venv/bin/pip" install --upgrade pip
"$REPO_DIR/.venv/bin/pip" install -r "$REPO_DIR/requirements.txt"

# Ensure data directory exists
mkdir -p "$REPO_DIR/data"

# Install systemd user service
mkdir -p "$HOME/.config/systemd/user"
cp "$REPO_DIR/deploy/eidolon.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable eidolon.service

echo ""
echo "=== Done ==="
echo "Next steps:"
echo "  1. Create $REPO_DIR/.env with your secrets"
echo "  2. Run: systemctl --user start eidolon"
echo "  3. Check: systemctl --user status eidolon"
echo "  4. Logs: journalctl --user -u eidolon -f"
