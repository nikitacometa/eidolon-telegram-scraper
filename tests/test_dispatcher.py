"""Tests for pipeline/dispatcher.py — alert delivery via Telegram bot."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.dispatcher import AlertDispatcher, _format_alert, _format_echo
from pipeline.models import DeliveryResult


def _response(status: int, *, retry_after: int | None = None) -> AsyncMock:
    """Build an aiohttp-style async context manager response."""
    response = AsyncMock()
    response.status = status
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    if retry_after is not None:
        response.json = AsyncMock(return_value={"parameters": {"retry_after": retry_after}})
    return response


@pytest.fixture
async def dispatcher() -> AsyncIterator[AlertDispatcher]:
    """Create an enabled dispatcher without relying on environment settings."""
    instance = AlertDispatcher(token="test-token", chat_id=12345)
    await instance.start()
    yield instance
    await instance.close()


class TestFormatAlert:
    def test_basic_format(self) -> None:
        """Should produce a readable alert message with HTML formatting."""
        msg = _format_alert(
            watcher_name="phangan-housing",
            chat_title="Phangan Expats",
            sender_name="Alice",
            text="Beautiful villa for rent, 3BR, pool",
            matched_keyword="villa",
            filter_level=1,
        )
        assert "Eidolon Alert" in msg
        assert "phangan-housing" in msg
        assert "Phangan Expats" in msg
        assert "Alice" in msg
        assert "villa" in msg
        assert "Beautiful villa for rent" in msg

    def test_long_message_truncated(self) -> None:
        """Messages longer than 500 chars should be truncated."""
        long_text = "x" * 600
        msg = _format_alert(
            watcher_name="test",
            chat_title="Test",
            sender_name="Bob",
            text=long_text,
            matched_keyword=None,
            filter_level=1,
        )
        assert "..." in msg
        assert len(msg) < 700

    def test_no_keyword(self) -> None:
        """Alert without matched keyword should not have keyword line."""
        msg = _format_alert(
            watcher_name="test",
            chat_title="Test",
            sender_name="Bob",
            text="Some text",
            matched_keyword=None,
            filter_level=1,
        )
        assert "Keyword" not in msg

    def test_escapes_all_untrusted_fields(self) -> None:
        """Telegram HTML markup from any dynamic field must remain inert."""
        msg = _format_alert(
            watcher_name='<watcher&">',
            chat_title="<b>Injected chat</b>",
            sender_name="Alice & Mallory",
            text='<a href="https://evil.example">click</a> & pay',
            matched_keyword="<rent>",
            filter_level=3,
        )

        assert "&lt;watcher&amp;&quot;&gt;" in msg
        assert "&lt;b&gt;Injected chat&lt;/b&gt;" in msg
        assert "Alice &amp; Mallory" in msg
        assert "&lt;rent&gt;" in msg
        assert '&lt;a href="https://evil.example"&gt;click&lt;/a&gt; &amp; pay' in msg
        assert "<b>Injected chat</b>" not in msg
        assert "<a href=" not in msg

    def test_echo_escapes_chat_sender_and_text(self) -> None:
        msg = _format_echo(
            chat_title="<i>Chat</i>",
            sender_name="A&B",
            text="<script>alert(1)</script>",
        )

        assert "&lt;i&gt;Chat&lt;/i&gt;" in msg
        assert "A&amp;B" in msg
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in msg
        assert "<script>" not in msg


class TestAlertDispatcher:
    def test_explicit_credentials_work_without_environment(self) -> None:
        """Injected credentials make tests and library use independent of .env."""
        with patch("pipeline.dispatcher.settings") as mock_settings:
            mock_settings.eidolon_bot_token = ""
            mock_settings.pantheon_bot_token = ""
            mock_settings.pantheon_chat_id = 0

            dispatcher = AlertDispatcher(token="injected-token", chat_id=987)

        assert dispatcher._enabled is True
        assert dispatcher._chat_id == 987
        assert "injected-token" in dispatcher._url

    async def test_send_alert_success(self, dispatcher: AlertDispatcher) -> None:
        """Should return True when bot API returns 200."""
        mock_resp = _response(200)

        with patch.object(dispatcher._session, "post", return_value=mock_resp):
            result = await dispatcher.send_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )
            assert result is True

    async def test_deliver_alert_returns_typed_success(
        self,
        dispatcher: AlertDispatcher,
    ) -> None:
        """Outbox callers should receive an explicit successful result."""
        with patch.object(dispatcher._session, "post", return_value=_response(200)):
            result = await dispatcher.deliver_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )

        assert result == DeliveryResult.success()

    async def test_deliver_alert_classifies_terminal_http_error(
        self,
        dispatcher: AlertDispatcher,
    ) -> None:
        """Non-rate-limit 4xx responses should not enter the durable retry queue."""
        post = MagicMock(return_value=_response(403))
        with patch.object(dispatcher._session, "post", post):
            result = await dispatcher.deliver_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )

        assert result == DeliveryResult(
            sent=False,
            retryable=False,
            error_code="telegram_http_403",
        )
        assert post.call_count == 1

    async def test_deliver_alert_preserves_retry_hint_after_local_retries(
        self,
    ) -> None:
        """A final 429 should expose retry timing for durable scheduling."""
        dispatcher = AlertDispatcher(
            token="test-token",
            chat_id=12345,
            max_attempts=1,
        )
        await dispatcher.start()
        try:
            with patch.object(
                dispatcher._session,
                "post",
                return_value=_response(429, retry_after=17),
            ):
                result = await dispatcher.deliver_alert(
                    watcher_name="test",
                    chat_title="Test Chat",
                    sender_name="Alice",
                    text="Test message",
                )
        finally:
            await dispatcher.close()

        assert result == DeliveryResult(
            sent=False,
            retryable=True,
            error_code="telegram_rate_limited",
            retry_after=17,
        )

    async def test_does_not_retry_client_error(
        self,
        dispatcher: AlertDispatcher,
    ) -> None:
        """A permanent 4xx response should fail immediately."""
        mock_resp = _response(403)
        post = MagicMock(return_value=mock_resp)
        sleep = AsyncMock()

        with (
            patch.object(dispatcher._session, "post", post),
            patch("pipeline.dispatcher.asyncio.sleep", sleep),
        ):
            result = await dispatcher.send_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )

        assert result is False
        assert post.call_count == 1
        sleep.assert_not_awaited()

    async def test_send_without_start(self) -> None:
        """Should return False if dispatcher not started."""
        dispatcher = AlertDispatcher(token="token", chat_id=123)
        result = await dispatcher.send_alert(
            watcher_name="test",
            chat_title="Test",
            sender_name="Bob",
            text="Hello",
        )
        assert result is False

    async def test_deliver_without_start_is_retryable(self) -> None:
        """A lifecycle race is recoverable and should retain a machine-readable cause."""
        dispatcher = AlertDispatcher(token="token", chat_id=123)

        result = await dispatcher.deliver_alert(
            watcher_name="test",
            chat_title="Test",
            sender_name="Bob",
            text="Hello",
        )

        assert result == DeliveryResult(
            sent=False,
            retryable=True,
            error_code="dispatcher_not_started",
            retry_after=1,
        )

    def test_eidolon_bot_token_preferred(self) -> None:
        """Should use eidolon_bot_token when available, falling back to pantheon."""
        with patch("pipeline.dispatcher.settings") as mock_settings:
            mock_settings.eidolon_bot_token = "eidolon-token-123"
            mock_settings.pantheon_bot_token = "pantheon-token-456"
            mock_settings.pantheon_chat_id = 12345
            dispatcher = AlertDispatcher()
            assert "eidolon-token-123" in dispatcher._url

    def test_fallback_to_pantheon_token(self) -> None:
        """Should fall back to pantheon_bot_token when eidolon_bot_token is empty."""
        with patch("pipeline.dispatcher.settings") as mock_settings:
            mock_settings.eidolon_bot_token = ""
            mock_settings.pantheon_bot_token = "pantheon-token-456"
            mock_settings.pantheon_chat_id = 12345
            dispatcher = AlertDispatcher()
            assert "pantheon-token-456" in dispatcher._url

    async def test_retries_server_error_with_backoff(
        self,
        dispatcher: AlertDispatcher,
    ) -> None:
        post = MagicMock(side_effect=[_response(500), _response(200)])
        sleep = AsyncMock()

        with (
            patch.object(dispatcher._session, "post", post),
            patch("pipeline.dispatcher.asyncio.sleep", sleep),
        ):
            result = await dispatcher.send_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )

        assert result is True
        assert post.call_count == 2
        sleep.assert_awaited_once_with(1)

    async def test_retries_timeout_with_backoff(
        self,
        dispatcher: AlertDispatcher,
    ) -> None:
        post = MagicMock(side_effect=[TimeoutError("slow"), _response(200)])
        sleep = AsyncMock()

        with (
            patch.object(dispatcher._session, "post", post),
            patch("pipeline.dispatcher.asyncio.sleep", sleep),
        ):
            result = await dispatcher.send_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )

        assert result is True
        assert post.call_count == 2
        sleep.assert_awaited_once_with(1)

    async def test_uses_telegram_retry_after(
        self,
        dispatcher: AlertDispatcher,
    ) -> None:
        post = MagicMock(side_effect=[_response(429, retry_after=7), _response(200)])
        sleep = AsyncMock()

        with (
            patch.object(dispatcher._session, "post", post),
            patch("pipeline.dispatcher.asyncio.sleep", sleep),
        ):
            result = await dispatcher.send_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )

        assert result is True
        sleep.assert_awaited_once_with(7)

    async def test_send_echo_uses_escaped_payload(
        self,
        dispatcher: AlertDispatcher,
    ) -> None:
        post = MagicMock(return_value=_response(200))
        with patch.object(dispatcher._session, "post", post):
            await dispatcher.send_echo(
                chat_title="<b>chat</b>",
                sender_name="A&B",
                text="<script>bad</script>",
            )

        payload = post.call_args.kwargs["json"]["text"]
        assert "&lt;b&gt;chat&lt;/b&gt;" in payload
        assert "A&amp;B" in payload
        assert "&lt;script&gt;bad&lt;/script&gt;" in payload

    async def test_send_summary_escapes_header_and_model_output(
        self,
        dispatcher: AlertDispatcher,
    ) -> None:
        post = MagicMock(return_value=_response(200))
        with patch.object(dispatcher._session, "post", post):
            result = await dispatcher.send_summary(
                watcher_name="<watcher>",
                summary='<a href="https://evil.example">offer</a> & more',
                date_str="<today>",
                message_count=1,
            )

        assert result is True
        payload = post.call_args.kwargs["json"]["text"]
        assert "&lt;watcher&gt;" in payload
        assert "&lt;today&gt;" in payload
        assert '&lt;a href="https://evil.example"&gt;offer&lt;/a&gt; &amp; more' in payload
        assert "<a href=" not in payload
