# Engineering Research and Rationale

> Updated 2026-07-17. Sources are primary documentation and current project
> advisories; prices and popularity rankings are intentionally omitted because
> they age quickly.

## What a Strong AI Backend Should Demonstrate

The target role combines LLM engineering with backend ownership. The strongest
repository signal is therefore not the number of AI frameworks imported, but a
system that is measurable, explainable, failure-aware, and reproducible.

| Industry guidance | Eidolon decision |
|---|---|
| Start with simple, composable workflows and add autonomy only when measurement justifies it ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents)). | Keep a deterministic rules → embeddings → LLM cascade instead of adding an agent framework as decoration. |
| Use task-specific, continuous evals with production-like data and explicit pass criteria ([OpenAI](https://developers.openai.com/api/docs/guides/evaluation-best-practices)). | Version an anonymized JSONL relevance set and gate precision/recall with `eidolon-eval`. |
| Structured outputs, guardrails, tracing, and usage metadata are first-class agent concerns ([OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)). | Validate model responses with Pydantic, separate trusted policy from untrusted Telegram content, and persist stage provenance. |
| GenAI telemetry should capture operation duration, model, and token usage ([OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)). | Use typed stage decisions and keep the schema ready for model/latency/token metrics without storing prompts by default. |
| AI risk management should continuously govern, map, measure, and manage risk ([NIST AI RMF](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)). | Combine offline eval gates, runtime funnel metrics, data retention, explicit degraded states, and documented failure policy. |
| Prompt injection, sensitive-data disclosure, supply-chain risk, and excessive agency are practical LLM threats ([OWASP GenAI](https://genai.owasp.org/2026/04/14/owasp-genai-exploit-round-up-report-q1-2026/)). | Treat messages as data, minimize stored payloads, scan dependencies, and keep Telegram writes outside the current scope. |

## Stack Decisions

- **Direct OpenAI SDK over LangChain.** There is one linear policy pipeline and no
  dynamic tool graph. Direct typed adapters reduce hidden state and make failure
  semantics testable.
- **SQLite WAL + embedded Chroma.** This is appropriate for a single-account,
  single-writer daemon. PostgreSQL/pgvector becomes justified when multiple workers,
  tenants, or remote querying require shared concurrency.
- **FastAPI as a separate read-only process.** Operational health and metrics are
  observable without starting a second MTProto client with the same sacred session.
- **At-least-once delivery.** A durable outbox recovers retryable failures. Exactly-once
  Telegram delivery is not claimed because the external Bot API has no idempotency key.

## Dependency Risk Note

The embedded Chroma dependency is held below the affected range in
[GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c).
`uv.lock`, Dependabot, and `pip-audit` make that exception visible and reviewable.

## Deliberate Non-Goals

Eidolon does not autonomously join groups, send replies, or expose the control plane
publicly. Those capabilities would require stronger identity, approval, rate-limit,
audit, and threat-model controls than a read-only portfolio daemon.
