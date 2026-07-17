#!/usr/bin/env bash
# First-time setup on a fresh VPS.
# Usage: bash deploy/install.sh
set -Eeuo pipefail

umask 077

readonly REPO_DIR="${REPO_DIR:-$HOME/eidolon-telegram-scraper}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

for command_name in uv systemctl; do
    command -v "$command_name" >/dev/null || fail "missing required command: $command_name"
done

[[ -d "$REPO_DIR/.git" ]] || fail "repository not found at $REPO_DIR"
[[ -f "$REPO_DIR/.env" ]] || fail "create $REPO_DIR/.env before installing the service"
[[ -f "$REPO_DIR/config/watchers.yml" ]] || {
    fail "copy config/watchers.example.yml to config/watchers.yml and configure it first"
}

printf 'Installing locked production dependencies...\n'
(
    cd "$REPO_DIR"
    uv sync --frozen --no-dev
)

mkdir -p "$REPO_DIR/data"
chmod 700 "$REPO_DIR/data"
chmod 600 "$REPO_DIR/.env"

mkdir -p "$HOME/.config/systemd/user"
cp "$REPO_DIR/deploy/eidolon.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable eidolon.service

printf '%s\n' \
    "Install complete." \
    "Start:  systemctl --user start eidolon" \
    "Status: systemctl --user status eidolon" \
    "Logs:   journalctl --user -u eidolon -f"
