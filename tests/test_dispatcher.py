"""Tests for pipeline/dispatcher.py — alert delivery via Telegram bot."""

from unittest.mock import AsyncMock, patch

import pytest

from pipeline.dispatcher import AlertDispatcher, _format_alert


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


class TestAlertDispatcher:
    async def test_send_alert_success(self) -> None:
        """Should return True when bot API returns 200."""
        dispatcher = AlertDispatcher()
        await dispatcher.start()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(dispatcher._session, "post", return_value=mock_resp):
            result = await dispatcher.send_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )
            assert result is True

        await dispatcher.close()

    async def test_send_alert_failure(self) -> None:
        """Should return False when bot API returns error."""
        dispatcher = AlertDispatcher()
        await dispatcher.start()

        mock_resp = AsyncMock()
        mock_resp.status = 403
        mock_resp.text = AsyncMock(return_value="Forbidden")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(dispatcher._session, "post", return_value=mock_resp):
            result = await dispatcher.send_alert(
                watcher_name="test",
                chat_title="Test Chat",
                sender_name="Alice",
                text="Test message",
            )
            assert result is False

        await dispatcher.close()

    async def test_send_without_start(self) -> None:
        """Should return False if dispatcher not started."""
        dispatcher = AlertDispatcher()
        result = await dispatcher.send_alert(
            watcher_name="test",
            chat_title="Test",
            sender_name="Bob",
            text="Hello",
        )
        assert result is False
