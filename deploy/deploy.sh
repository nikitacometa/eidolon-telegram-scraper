#!/usr/bin/env bash
# Deploy latest changes to the VPS.
# Usage: bash deploy/deploy.sh
set -euo pipefail

VPS_HOST="metaflexer@72.60.104.156"
REMOTE_DIR="eidolon-telegram"

echo "=== Deploying Eidolon ==="

# Push local changes
echo "Pushing to origin..."
git push origin main

# Pull on remote, install deps, restart
echo "Updating on VPS..."
ssh "$VPS_HOST" bash -s <<'REMOTE'
set -euo pipefail
cd ~/eidolon-telegram
git pull origin main
.venv/bin/pip install -q -r requirements.txt
systemctl --user restart eidolon
sleep 2
systemctl --user status eidolon --no-pager
REMOTE

echo "=== Deploy complete ==="
