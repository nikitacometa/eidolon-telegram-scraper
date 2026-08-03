"""Where the structured LLM calls actually run.

Two engines answer the same question with the same schema.

``api`` calls ``api.openai.com`` in-process and is billed per token. ``subscription``
shells out to the Codex CLI, which authenticates with the owner's ChatGPT
subscription and therefore costs no money — only quota.

They are not interchangeable in every respect, and the differences are the
reason this is a setting rather than a decision baked into the pipeline:

* **Latency.** The API answers in about 1.5 s. The CLI is a sandboxed coding
  agent that boots per call; measured on this machine, 17-18 s. Fine for a
  monitoring pipeline whose alerts are minutes-fresh, useless for anything
  interactive.
* **Overhead.** Each CLI call carries roughly 21,500 tokens of its own context
  before it sees the message — measured, not estimated. Against a quota shared
  with the owner's assistant, that is the real cost, and it scales with call
  count rather than with message size.
* **Embeddings do not exist here.** Neither CLI exposes an embedding endpoint,
  so Level 2 stays on the API regardless of this setting. There is no version of
  this module that changes that.

The subprocess never runs on the ingest path and never raises into it: a failure
returns a degraded decision, which the watcher's own policy then judges.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from config.settings import settings

logger = logging.getLogger(__name__)

# Measured on this machine, 2026-08-03: a one-shot `codex exec` classification
# of a 300-character message returned in 17-18 s and reported 21,571 tokens.
# The timeout is deliberately far above that — a slow answer is worth waiting
# for, a hung subprocess is not.
SUBPROCESS_TIMEOUT_SECONDS = 120


class EngineError(RuntimeError):
    """The engine could not produce an answer. The caller decides what that means."""


@dataclass(frozen=True, slots=True)
class EngineResult:
    """One structured answer plus what it cost to get."""

    payload: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class StructuredEngine(Protocol):
    """Anything that can turn a prompt into a schema-shaped dict."""

    name: str

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        timeout_seconds: float,
    ) -> EngineResult: ...


class SubscriptionEngine:
    """Runs the call through the Codex CLI on the owner's ChatGPT subscription.

    ``--output-schema`` is the reason this is the CLI worth using rather than the
    OpenClaw one: the schema is enforced where the answer is produced, so a
    malformed reply is the provider's problem rather than a parse error here.
    """

    name = "subscription"

    def __init__(
        self,
        *,
        binary: str = "codex",
        effort: str = "low",
        timeout: float = SUBPROCESS_TIMEOUT_SECONDS,
    ) -> None:
        self._binary = binary
        self._effort = effort
        self._timeout = timeout

    def available(self) -> bool:
        return shutil.which(self._binary) is not None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        timeout_seconds: float,
    ) -> EngineResult:
        if not self.available():
            raise EngineError(f"{self._binary} is not on PATH")

        # A private working directory per call: the CLI writes its output file
        # and its own scratch there, and two concurrent classifications must not
        # meet each other's files.
        with tempfile.TemporaryDirectory(prefix="eidolon-engine-") as workdir:
            work = Path(workdir)
            schema_path = work / "schema.json"
            out_path = work / "out.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")

            argv = [
                self._binary,
                "exec",
                "-m",
                model,
                "-c",
                f"model_reasoning_effort={self._effort}",
                "--skip-git-repo-check",
                "-C",
                str(work),
                "-s",
                "read-only",
                "--output-schema",
                str(schema_path),
                "-o",
                str(out_path),
                f"{system}\n\n{user}",
            ]

            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            budget = min(timeout_seconds, self._timeout)
            try:
                async with asyncio.timeout(budget):
                    _, stderr = await process.communicate()
            except TimeoutError:
                # This runs inside the daemon that owns the Telegram session; a
                # subprocess that never returns would hold a pipeline worker for
                # the life of the process.
                process.kill()
                await process.wait()
                raise EngineError(f"{self._binary} exceeded {budget}s") from None

            if process.returncode != 0:
                detail = (stderr or b"").decode("utf-8", "replace").strip()[-300:]
                raise EngineError(f"{self._binary} exited {process.returncode}: {detail}")
            if not out_path.exists():
                raise EngineError(f"{self._binary} wrote no output file")
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise EngineError(f"{self._binary} output was not JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise EngineError("engine returned a non-object payload")

        # The CLI does not report usage in a machine-readable form, and inventing
        # a number here would poison the cost accounting in pipeline_runs. Zero
        # means "not measured", which the reader can tell apart from "free".
        return EngineResult(payload=payload, model=model)


class OpenAIEngine:
    """Calls the API in-process with a JSON-schema response format."""

    name = "api"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _require_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        timeout_seconds: float,
    ) -> EngineResult:
        response = await self._require_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": schema, "strict": True},
            },
            timeout=timeout_seconds,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise EngineError("empty response")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise EngineError(f"response was not JSON: {exc}") from exc
        usage = getattr(response, "usage", None)
        return EngineResult(
            payload=payload,
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


def build_engine(name: str | None = None) -> StructuredEngine:
    """Return the configured engine, falling back loudly rather than silently."""
    choice = (name or settings.llm_engine).strip().lower()
    if choice == "subscription":
        engine = SubscriptionEngine(effort=settings.subscription_effort)
        if engine.available():
            return engine
        # Falling back is right — a missing CLI must not stop the pipeline — but
        # it changes who pays, so it is never allowed to happen quietly.
        logger.warning(
            "llm_engine=subscription but the codex CLI is not on PATH; "
            "falling back to the paid API engine"
        )
        return OpenAIEngine()
    if choice != "api":
        logger.warning("Unknown llm_engine %r; using the API engine", choice)
    return OpenAIEngine()
