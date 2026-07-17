#!/usr/bin/env bash
# Deploy one already-pushed immutable revision to the VPS.
# Usage: CONFIRM_DEPLOY=yes VPS_HOST=my-host bash deploy/deploy.sh
set -Eeuo pipefail

: "${VPS_HOST:?Set VPS_HOST to an SSH alias or user@host}"
: "${CONFIRM_DEPLOY:?Set CONFIRM_DEPLOY=yes after reviewing the target}"
[[ "$CONFIRM_DEPLOY" == "yes" ]] || {
    printf 'ERROR: CONFIRM_DEPLOY must equal yes\n' >&2
    exit 1
}

readonly REMOTE_DIR="${REMOTE_DIR:-eidolon-telegram-scraper}"
readonly REVISION="${REVISION:-$(git rev-parse HEAD)}"

[[ -z "$(git status --porcelain)" ]] || {
    printf 'ERROR: deploy from a clean working tree\n' >&2
    exit 1
}
git cat-file -e "${REVISION}^{commit}"
git fetch --quiet origin main
git merge-base --is-ancestor "$REVISION" origin/main || {
    printf 'ERROR: revision %s is not pushed to origin/main\n' "$REVISION" >&2
    exit 1
}

printf 'Deploying revision %s to %s:%s\n' "$REVISION" "$VPS_HOST" "$REMOTE_DIR"
ssh "$VPS_HOST" bash -s -- "$REMOTE_DIR" "$REVISION" <<'REMOTE'
set -Eeuo pipefail
remote_dir=$1
revision=$2
cd "$HOME/$remote_dir"

command -v uv >/dev/null
previous_revision=$(git rev-parse HEAD)
rollback() {
    printf 'Deployment failed; restoring %s\n' "$previous_revision" >&2
    git checkout --detach "$previous_revision"
    uv sync --frozen --no-dev
    systemctl --user restart eidolon
}
trap rollback ERR

git fetch --quiet origin
git cat-file -e "${revision}^{commit}"
git checkout --detach "$revision"
uv sync --frozen --no-dev
systemctl --user restart eidolon
systemctl --user is-active --quiet eidolon
trap - ERR
REMOTE

printf 'Deployment complete: %s\n' "$REVISION"
