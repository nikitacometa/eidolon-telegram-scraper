# Security Policy

## Supported Versions

Security fixes are applied to the current release line only.

| Release line | Supported |
| --- | --- |
| `0.2.x` | Yes |
| `< 0.2` | No |

## Reporting a Vulnerability

Do not disclose vulnerabilities in a public issue, discussion, pull request, or log
excerpt. Open the repository's **Security** tab and select **Report a vulnerability** to
create a private GitHub advisory. Include the affected version, impact, reproduction
steps, and a minimal proof of concept. Redact Telegram sessions, API keys, message
contents, chat IDs, and personal data.

If private reporting is unavailable, contact the maintainer through their GitHub profile
and request a private channel without including exploit details. Please allow time to
validate and coordinate a fix before public disclosure.

## Credentials and Account Safety

Treat `TELEGRAM_SESSION_STRING` as a full account credential, not as ordinary
configuration. Use a dedicated Telegram account with the minimum required group access;
never use a primary personal account. Run only one worker for each session string.

Keep Telegram, OpenAI, and bot credentials in `.env` or a deployment secret store. Never
bake them into an image, commit them, paste them into issues, or emit them in logs. Keep
`.env` mode `0600` and the data directory mode `0700`. If exposure is suspected, revoke
the Telegram session and rotate every affected provider token immediately.

`config/watchers.yml` may reveal private chat IDs and monitoring objectives. It is local
configuration and must remain untracked; commit only the redacted example file.

## Data Privacy and Retention

SQLite stores message text, sender and chat metadata, pipeline decisions, and alert
delivery state. L2 sends message text and watcher examples to OpenAI for embeddings. L3
sends message text plus the watcher description/prompt to OpenAI; enabled summaries send
selected message fields. Do not put secrets, private identifiers, or unnecessary PII in
watcher objectives or examples. Embedded Chroma stores reference vectors, opaque IDs, and
positive/negative labels; reference plaintext remains in local watcher configuration and
process memory rather than Chroma documents. `STORE_RAW_TELEGRAM_JSON` is disabled by
default, but disabling it does not disable normal message-text storage.

`RETENTION_DAYS` defaults to 30 days. The worker purges at startup and repeats the sweep
every `RETENTION_SWEEP_HOURS` (24 by default). Pending pipeline jobs and pending alert
deliveries are exempt until they become terminal so retention cannot silently destroy
recoverable work; monitor backlog age because this can extend storage beyond the nominal
window. Failed jobs remain visible through `/stats` and become retention-eligible.
Encrypt and access-control host volumes and backups, and expire backups under the same
retention policy.

## Control-Plane Exposure

The FastAPI control plane has no application authentication. Bind it to loopback, as the
provided Compose configuration does, or place it behind a reverse proxy that enforces
TLS and authentication. Before any external exposure, also configure trusted hosts,
request-size and rate limits, trusted proxy addresses, and protection or disabling of
`/docs`, `/redoc`, and `/openapi.json`. Never publish the container's port directly.

## Deployment Checklist

- Use a dedicated least-privilege OS user and Telegram account.
- Inject secrets at runtime; verify `.env`, sessions, databases, and backups are excluded
  from Git and the image build context.
- Keep exactly one worker replica per Telethon session.
- Mount watcher configuration read-only and persistent data only where required.
- Expose the control plane only on loopback or through authenticated TLS.
- Review retention, backup restoration, and credential-rotation procedures.
- Run the locked test, lint, dependency-audit, and container-build checks before release.
