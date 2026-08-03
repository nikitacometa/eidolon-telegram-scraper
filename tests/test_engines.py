"""Tests for pipeline/engines.py — where the structured LLM calls run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.engines import (
    EngineError,
    OpenAIEngine,
    SubscriptionEngine,
    build_engine,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"relevant": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["relevant", "reason"],
}


def _fake_codex(tmp_path: Path, *, body: str, exit_code: int = 0, sleep: float = 0.0) -> str:
    """A stand-in for the CLI that honours -o and --output-schema."""
    script = tmp_path / "fake-codex"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        f"time.sleep({sleep})\n"
        "argv = sys.argv[1:]\n"
        "out = argv[argv.index('-o') + 1] if '-o' in argv else None\n"
        f"code = {exit_code}\n"
        "if code == 0 and out is not None:\n"
        f"    open(out, 'w').write({body!r})\n"
        "else:\n"
        "    sys.stderr.write('fake failure detail')\n"
        "sys.exit(code)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


class TestSubscriptionEngine:
    async def test_a_schema_shaped_answer_comes_back_as_a_dict(self, tmp_path: Path) -> None:
        engine = SubscriptionEngine(
            binary=_fake_codex(tmp_path, body=json.dumps({"relevant": True, "reason": "quiz"}))
        )
        result = await engine.complete(
            system="s", user="u", schema=SCHEMA, model="gpt-5.6-luna", timeout_seconds=30
        )
        assert result.payload == {"relevant": True, "reason": "quiz"}
        assert result.model == "gpt-5.6-luna"

    async def test_usage_is_reported_as_unmeasured_rather_than_as_free(
        self, tmp_path: Path
    ) -> None:
        # The CLI does not report usage; inventing a number would corrupt the
        # per-call token accounting that pipeline_runs stores.
        engine = SubscriptionEngine(
            binary=_fake_codex(tmp_path, body=json.dumps({"relevant": False, "reason": "x"}))
        )
        result = await engine.complete(
            system="s", user="u", schema=SCHEMA, model="m", timeout_seconds=30
        )
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    async def test_a_nonzero_exit_is_an_engine_error_not_a_verdict(self, tmp_path: Path) -> None:
        engine = SubscriptionEngine(binary=_fake_codex(tmp_path, body="", exit_code=3))
        with pytest.raises(EngineError, match="exited 3"):
            await engine.complete(
                system="s", user="u", schema=SCHEMA, model="m", timeout_seconds=30
            )

    async def test_unparseable_output_is_an_engine_error(self, tmp_path: Path) -> None:
        engine = SubscriptionEngine(binary=_fake_codex(tmp_path, body="not json at all"))
        with pytest.raises(EngineError, match="not JSON"):
            await engine.complete(
                system="s", user="u", schema=SCHEMA, model="m", timeout_seconds=30
            )

    async def test_a_json_array_is_rejected(self, tmp_path: Path) -> None:
        engine = SubscriptionEngine(binary=_fake_codex(tmp_path, body="[1, 2, 3]"))
        with pytest.raises(EngineError, match="non-object"):
            await engine.complete(
                system="s", user="u", schema=SCHEMA, model="m", timeout_seconds=30
            )

    async def test_a_hung_subprocess_is_killed_rather_than_waited_on(self, tmp_path: Path) -> None:
        # This runs inside the daemon that owns the Telegram session. A call that
        # never returns would hold a pipeline worker for the process lifetime.
        engine = SubscriptionEngine(binary=_fake_codex(tmp_path, body="{}", sleep=30), timeout=1.0)
        with pytest.raises(EngineError, match="exceeded"):
            await engine.complete(
                system="s", user="u", schema=SCHEMA, model="m", timeout_seconds=1.0
            )

    async def test_a_missing_binary_is_reported_not_crashed(self) -> None:
        engine = SubscriptionEngine(binary="definitely-not-a-real-binary-xyz")
        assert not engine.available()
        with pytest.raises(EngineError, match="not on PATH"):
            await engine.complete(system="s", user="u", schema=SCHEMA, model="m", timeout_seconds=5)

    async def test_concurrent_calls_do_not_share_an_output_file(self, tmp_path: Path) -> None:
        # Each call gets its own working directory; without that, two
        # classifications running together would read each other's answers.
        import asyncio

        engine = SubscriptionEngine(
            binary=_fake_codex(tmp_path, body=json.dumps({"relevant": True, "reason": "r"}))
        )
        results = await asyncio.gather(
            *(
                engine.complete(
                    system="s", user=f"u{i}", schema=SCHEMA, model="m", timeout_seconds=30
                )
                for i in range(6)
            )
        )
        assert all(r.payload["relevant"] for r in results)


class TestEngineSelection:
    def test_the_api_engine_is_the_default(self) -> None:
        assert build_engine("api").name == "api"

    def test_an_unknown_name_falls_back_to_the_api(self) -> None:
        assert build_engine("nonsense").name == "api"

    def test_subscription_falls_back_when_the_cli_is_absent(self, monkeypatch: Any) -> None:
        # Falling back is correct — a missing CLI must not stop monitoring — but
        # it silently changes who pays, so the warning matters as much as the
        # fallback.
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert build_engine("subscription").name == "api"

    def test_subscription_is_selected_when_the_cli_exists(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/codex")
        assert build_engine("subscription").name == "subscription"


class TestOpenAIEngine:
    async def test_an_empty_response_is_an_engine_error(self) -> None:
        class _Msg:
            content = ""

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = None

        class _Completions:
            async def create(self, **_: Any) -> Any:
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        with pytest.raises(EngineError, match="empty response"):
            await OpenAIEngine(client=_Client()).complete(
                system="s", user="u", schema=SCHEMA, model="m", timeout_seconds=5
            )

    async def test_token_usage_is_carried_through(self) -> None:
        class _Msg:
            content = '{"relevant": true, "reason": "ok"}'

        class _Choice:
            message = _Msg()

        class _Usage:
            prompt_tokens = 700
            completion_tokens = 42

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        class _Completions:
            async def create(self, **_: Any) -> Any:
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        result = await OpenAIEngine(client=_Client()).complete(
            system="s", user="u", schema=SCHEMA, model="m", timeout_seconds=5
        )
        assert (result.input_tokens, result.output_tokens) == (700, 42)
        assert result.payload["relevant"] is True
