# Eidolon — Technology Research

> Date: 2026-03-04

## Telegram Client Libraries

| Library | Stars | Status | Last Release | Recommendation |
|---------|-------|--------|-------------|----------------|
| **Telethon** | 11.9k | Active (moved to Codeberg) | v1.42.0 (Nov 2025) | **Use this** |
| Pyrogram | ~8k | **Archived Dec 2024** | v2.0.106 | Dead, do not use |
| Hydrogram | 214 | Moderate | v0.2.0 (Jun 2024) | Pyrogram fork, backup option |
| PyroFork | 277 | Active | Active (126 tags) | Pyrogram fork, alternative |
| python-telegram-bot | ~27k | Active | v22.0 | Bot API only, not for userbots |

**Key facts:**
- Telethon moved from GitHub to Codeberg: https://codeberg.org/Lonami/Telethon
- v2 alpha (2.0.0a0) exists but is a complete rewrite — not production-ready
- 1.79M monthly PyPI downloads — dominant in the space
- Built-in FloodWaitError handling (`flood_sleep_threshold`)

## Existing OSS Projects

| Project | Stars | Stack | LLM |
|---------|-------|-------|-----|
| [telegram-chat-summarizer](https://github.com/dev0x13/telegram-chat-summarizer) | 91 | Telethon + LangChain | GPT-4 Turbo |
| [telegram-summary-bot](https://github.com/asukaminato0721/telegram-summary-bot) | 182 | TypeScript, Cloudflare | Gemini 2.5 Flash |
| [elizaOS/discord-summarizer](https://github.com/elizaOS/discord-summarizer) | 90 | Python, LangChain | Ollama (local) |

**Conclusion:** No production-quality "userbot + LLM monitoring" solution exists. This is a gap.

## Architecture Decision

Chose **Pattern 2: Buffer + Periodic Processing** over direct processing or message queue.

Three-level filter pipeline:
1. **Rule-based** (free) — keywords, regex, sender lists → drops 80-90%
2. **Embedding similarity** (~$0.02/1M tokens) — cosine similarity vs reference vectors → drops 50-70%
3. **LLM classification** (expensive) — only for filtered candidates

## Cost Model

| Component | Cost/1M tokens | Use case |
|-----------|---------------|----------|
| OpenAI text-embedding-3-small | $0.02 input | Pre-filtering |
| Claude Haiku 3.5 | $0.80 input / $4 output | Quick classification |
| Claude Sonnet (Batch API) | $1.50 input / $7.50 output | Summaries (50% off) |
| GPT-4o-mini | $0.15 input / $0.60 output | Cheapest alternative |

## Telegram Account Safety

| Action | Risk Level |
|--------|-----------|
| Passive message reading | Very low |
| Sending < 1 msg/min | Low |
| Sending to > 20 chats/min | High |
| Mass group joining | High |
| Identical messages to many | Critical |

**Best practices:** dedicated phone number, StringSession (reuse), single client instance, human-like delays.

## Sources

- [Telethon Codeberg](https://codeberg.org/Lonami/Telethon)
- [Telethon docs v1.42](https://docs.telethon.dev/en/stable/)
- [telegram-chat-summarizer](https://github.com/dev0x13/telegram-chat-summarizer)
- [Anthropic Batch API](https://docs.anthropic.com/en/docs/build-with-claude/message-batches)
- [GPTCache](https://github.com/zilliztech/GPTCache)
- [SentryDock](https://www.sentrydock.com/) — commercial analog
- [PyPI stats Telethon](https://www.pypistats.org/packages/telethon)
