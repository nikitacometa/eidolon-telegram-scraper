# Eidolon

> *εἴδωλον (eidōlon) — the phantom double of a living person in Greek mythology*

**AI-powered digital presence for Telegram.** Connects to a real account via MTProto, monitors group chats, filters messages through a multi-level intelligence pipeline, and delivers actionable alerts.

Not a bot. A phantom.

## What It Does

Eidolon silently monitors Telegram group chats and runs every message through a three-level filter:

| Level | Method | Cost | Drop Rate |
|-------|--------|------|-----------|
| **1. Rules** | Keywords, regex, sender lists | Free | ~80-90% |
| **2. Embeddings** | Cosine similarity vs reference vectors | ~$0.02/1M tokens | ~50-70% |
| **3. LLM** | GPT-4.1-mini classification + GPT-4.1 analysis | Per-token | Final filter |

Messages that survive all levels trigger instant alerts via Telegram bot — with extracted data, relevance scores, and optional LLM summaries.

## Architecture

```
Telegram Groups (MTProto)
    ↓ events.NewMessage
  Rule Filter → Embedding Filter → LLM Analysis
    ↓                                    ↓
  dropped                         Alert Dispatcher
                                        ↓
                              Telegram / Email / Webhook
                                        ↓
                                SQLite (full audit trail)
```

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/nikitacometa/eidolon-telegram.git
cd eidolon-telegram
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in Telegram API credentials (https://my.telegram.org)
# Fill in OpenAI API key

# 3. Generate Telegram session
python3 auth.py

# 4. Configure watchers
# Edit config/watchers.yml — define which chats to monitor and what to look for

# 5. Run
python3 main.py
```

## Watchers

Watchers are declarative rules in `config/watchers.yml`:

```yaml
watchers:
  - name: phangan-housing
    chats: [-100123456789]
    rules:
      keywords: [house, villa, rent, сдаю, дом, аренда]
      keywords_negative: [ищу, looking for]
      min_length: 20
    alert: immediate
    llm_level: 3
    prompt: |
      Is someone offering a house/villa for rent?
      Extract: type, location, price, duration, contact.
```

Each watcher defines: **what chats** to listen to, **what rules** to apply, **how urgently** to alert, and **how deep** the AI analysis should go.

## Stack

| Component | Technology |
|-----------|-----------|
| Runtime | Python 3.12+ |
| Telegram | Telethon v1.42 (MTProto, userbot) |
| LLM | OpenAI GPT-4.1-mini / GPT-4.1 |
| Embeddings | OpenAI text-embedding-3-small |
| Storage | SQLite + aiosqlite |
| Vector Store | ChromaDB (embedded) |
| Config | Pydantic Settings + YAML |

## Roadmap

1. **MVP** — Read-only monitor → keyword filter → Telegram alerts
2. **Smart Filter** — Embedding pre-filter + LLM classification
3. **Summarizer** — Periodic chat digests (morning/evening)
4. **Responder** — Reply to patterns with user approval
5. **Autonomous** — Join chats, manage personas, initiate conversations

## License

MIT
