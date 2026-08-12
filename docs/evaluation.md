# Evaluation Lab

Eidolon treats relevance as a measurable product behavior, not a prompt demo. The
evaluation runner replays labeled, anonymized messages through the same rule, embedding,
intent-policy, and structured-LLM components used by the worker.

## Dataset Roles

| Dataset | Role | Cases | Notes |
| --- | --- | ---: | --- |
| `relevance-calibration-v1.jsonl` | L2 calibration | 24 | EN/RU examples used to choose a recall-oriented semantic gate |
| `relevance-v1.jsonl` | Development validation | 20 | Stable regression set used while building the pipeline |
| `relevance-holdout-v2.jsonl` | Adversarial development set | 32 | Balanced set that exposed deterministic-gate recall loss |
| `relevance-holdout-v3.jsonl` | Final blind holdout | 40 | Independently authored, balanced by label and language; not used for tuning |

Current v0.2 artifacts include SHA-256 hashes for the dataset and watcher config, the
prompt version, models, intent and degradation policy, threshold, margin, per-case
stopping stage, aggregate latency/token usage, and machine-readable stage errors.
Historical blind artifacts retain their original config hash; an omitted
`degraded_policy` denotes the legacy availability-first `accept` behavior and is handled
explicitly by compatibility code.

## Reproduce

The offline Level 1 run is deterministic and needs no credentials:

```bash
uv run eidolon-eval --level 1 --output docs/evaluation-baseline.json
```

Online runs require `OPENAI_API_KEY` and incur provider usage:

```bash
uv run eidolon-eval \
  --level 2 \
  --dataset evals/data/relevance-calibration-v1.jsonl \
  --output docs/evaluation-calibration-level2-2026-07-17.json

uv run eidolon-calibrate \
  docs/evaluation-calibration-level2-2026-07-17.json \
  --output docs/embedding-calibration-2026-07-17.json

uv run eidolon-eval \
  --level 3 \
  --dataset evals/data/relevance-holdout-v3.jsonl \
  --output docs/evaluation-holdout-v3-post-review-2026-07-17.json
```

## Current Results

| Run | Precision | Recall | F1 | Degraded | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| L1 validation, 20 cases | 0.889 | 1.000 | 0.941 | 0 | pass |
| L2 calibration, 24 cases | 0.833 | 1.000 | 0.909 | 0 | pass |
| L3 validation, 20 cases | 1.000 | 1.000 | 1.000 | 0 | pass |
| Initial L3 blind holdout, 40 cases | 1.000 | 0.800 | 0.889 | 1 | **fail closed** |
| Post-review frozen-set regression, 40 cases | 1.000 | 0.750 | 0.857 | 1 | **fail closed** |

The default gate requires precision and recall of at least `0.80` and zero degraded
predictions. The initial blind run met the quality thresholds but failed because one
provider response returned non-verbatim evidence. That run used the then-current
availability-first fallback. A subsequent security review changed watcher defaults to
`degraded_policy: reject`; `accept` now requires an explicit policy choice.

Replaying the already-inspected frozen set after that security change is a regression
check, not a new blind measurement. It converts the degraded positive into a fifth false
negative, so recall is `0.75`; no prompt or threshold was tuned against this result. The
next milestone is an uncertainty-routing experiment on a development corpus, followed by
a newly authored blind set. Reported token and provider-call totals cover per-message
inference; one-time reference index seeding is excluded.

## Entity Retrieval Gate

Open-taxonomy retrieval has a separate 48-query JSONL contract. Forty-two cases are
verified against the anonymized raw-message fixture; six hookah cases are explicitly
`pending_evidence` and are reported but not scored. The pending marker is not a positive
label: production evidence currently shows marketplace ads, not a named lounge, and the
owner's live sweep must decide whether those cases become verified `must_be_empty` cases.

The deterministic CI command makes no provider calls:

```bash
uv run eidolon-place-eval \
  --dataset evals/data/place-retrieval-golden-v1.jsonl \
  --fixture evals/data/place-retrieval-fixture-v1.db \
  --query-vectors evals/data/place-query-vectors-v1.npz \
  --k 5 \
  --fail-on-regression docs/place-retrieval-baseline-v1.json
```

The deployment smoke is lexical-only and has a five-second budget:

```bash
uv run eidolon-place-eval --smoke --db data/eidolon_search.db --timeout 5
```

The smoke checks migration invariants, old/new name lookup, SYNCHØUSE folding, terminal
retrieval, honest empty behavior, FTS row parity, active prompt version, and descriptor
embedding backlog. It reports evidence-dependent checks as pending; it does not turn
missing labels into green results and does not replace the full gate.
