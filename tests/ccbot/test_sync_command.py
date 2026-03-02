"""Tests for /sync command — state audit and cleanup."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.handlers.callback_data import CB_SYNC_DISMISS, CB_SYNC_FIX
from ccbot.handlers.sync_command import (
    _format_report,
    handle_sync_dismiss,
    handle_sync_fix,
    sync_command,
)
from ccbot.session import AuditIssue, AuditResult


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch("ccbot.handlers.sync_command.session_manager") as mock_sm,
        patch("ccbot.handlers.sync_command.tmux_manager") as mock_tm,
        patch("ccbot.handlers.sync_command.config") as mock_cfg,
    ):
        mock_sm.audit_state.return_value = AuditResult(
            issues=[], total_bindings=0, live_binding_count=0
        )
        mock_sm.iter_thread_bindings.return_value = []
        mock_sm.window_states = {}
        mock_tm.list_windows = AsyncMock(return_value=[])
        mock_cfg.is_user_allowed.return_value = True
        yield mock_sm, mock_tm, mock_cfg


class TestBuildReport:
    @pytest.mark.parametrize(
        ("audit", "expected_text"),
        [
            pytest.param(
                AuditResult(issues=[], total_bindings=3, live_binding_count=3),
                "3 topics bound, all windows alive",
                id="all-alive",
            ),
            pytest.param(
                AuditResult(issues=[], total_bindings=0, live_binding_count=0),
                "No topic bindings",
                id="no-bindings",
            ),
            pytest.param(
                AuditResult(
                    issues=[
                        AuditIssue("ghost_binding", "thread 42 → @7", fixable=False)
                    ],
                    total_bindings=3,
                    live_binding_count=2,
                ),
                "ghost binding",
                id="ghost-only",
            ),
        ],
    )
    def test_no_keyboard_cases(self, audit: AuditResult, expected_text: str) -> None:
        text, keyboard = _format_report(audit)
        assert expected_text in text
        assert keyboard is None

    def test_fixable_issues_show_fix_button(self) -> None:
        audit = AuditResult(
            issues=[
                AuditIssue("orphaned_display_name", "@7 (old)", fixable=True),
            ],
            total_bindings=3,
            live_binding_count=3,
        )
        text, keyboard = _format_report(audit)
        assert "1 orphaned display name" in text
        assert keyboard is not None
        data = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]
        assert CB_SYNC_FIX in data
        assert CB_SYNC_DISMISS in data
        assert "Fix 1 issue" in keyboard.inline_keyboard[0][0].text

    def test_fixed_mode_header(self) -> None:
        audit = AuditResult(issues=[], total_bindings=3, live_binding_count=3)
        text, _keyboard = _format_report(audit, fixed_count=2)
        assert "\u2705 Fixed 2 issues" in text

    def test_multiple_fixable_issues(self) -> None:
        audit = AuditResult(
            issues=[
                AuditIssue("orphaned_display_name", "@7 (old)", fixable=True),
                AuditIssue("stale_offset", "user 100, window @9", fixable=True),
                AuditIssue(
                    "display_name_drift", "@1: stored='a' tmux='b'", fixable=True
                ),
            ],
            total_bindings=3,
            live_binding_count=3,
        )
        _text, keyboard = _format_report(audit)
        assert keyboard is not None
        assert "Fix 3 issues" in keyboard.inline_keyboard[0][0].text

    def test_clean_state_shows_all_clear(self) -> None:
        audit = AuditResult(issues=[], total_bindings=3, live_binding_count=3)
        text, _keyboard = _format_report(audit)
        assert "No orphaned entries" in text


class TestSyncDismiss:
    async def test_dismiss_removes_keyboard(self, _patch_deps) -> None:
        query = AsyncMock()
        msg = MagicMock()
        msg.text = "some report text"
        query.message = msg

        with patch("ccbot.handlers.sync_command.safe_edit") as mock_edit:
            await handle_sync_dismiss(query)
            mock_edit.assert_called_once()
            assert mock_edit.call_args[0][1] == "some report text"
            assert mock_edit.call_args.kwargs.get("reply_markup") is None

    async def test_dismiss_fallback_when_no_text(self, _patch_deps) -> None:
        query = AsyncMock()
        query.message = None

        with patch("ccbot.handlers.sync_command.safe_edit") as mock_edit:
            await handle_sync_dismiss(query)
            mock_edit.assert_called_once()
            assert mock_edit.call_args[0][1] == "Dismissed"


class TestSyncCommand:
    async def test_unauthorized_user_rejected(self, _patch_deps) -> None:
        _, _, mock_cfg = _patch_deps
        mock_cfg.is_user_allowed.return_value = False

        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()

        with patch("ccbot.handlers.sync_command.safe_reply") as mock_reply:
            await sync_command(update, MagicMock())
            mock_reply.assert_called_once()
            assert "not authorized" in mock_reply.call_args[0][1]

    async def test_no_user_returns_early(self, _patch_deps) -> None:
        update = MagicMock()
        update.effective_user = None
        update.message = AsyncMock()

        with patch("ccbot.handlers.sync_command.safe_reply") as mock_reply:
            await sync_command(update, MagicMock())
            mock_reply.assert_not_called()

    async def test_calls_audit_and_replies(self, _patch_deps) -> None:
        mock_sm, _, _ = _patch_deps
        mock_sm.audit_state.return_value = AuditResult(
            issues=[], total_bindings=2, live_binding_count=2
        )

        update = MagicMock()
        update.effective_user = MagicMock(id=100)
        update.message = AsyncMock()

        with patch("ccbot.handlers.sync_command.safe_reply") as mock_reply:
            await sync_command(update, MagicMock())
            mock_reply.assert_called_once()
            mock_sm.audit_state.assert_called_once()
            assert "2 topics bound" in mock_reply.call_args[0][1]


class TestSyncFix:
    async def test_fix_runs_cleanup_and_re_audits(self, _patch_deps) -> None:
        mock_sm, _, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue("orphaned_display_name", "@7 (old)", fixable=True),
                ],
                total_bindings=2,
                live_binding_count=2,
            ),
            AuditResult(issues=[], total_bindings=2, live_binding_count=2),
        ]

        query = AsyncMock()

        with patch("ccbot.handlers.sync_command.safe_edit") as mock_edit:
            await handle_sync_fix(query)
            mock_sm.sync_display_names.assert_called_once_with([])
            mock_sm.prune_stale_state.assert_called_once_with(set())
            mock_sm.prune_session_map.assert_called_once_with(set())
            mock_sm.prune_stale_window_states.assert_called_once_with(set())
            mock_sm.prune_stale_offsets.assert_called_once_with(set())
            assert mock_sm.audit_state.call_count == 2
            mock_edit.assert_called_once()
            assert "\u2705 Fixed 1 issue" in mock_edit.call_args[0][1]

    async def test_fix_computes_actual_fixed_count(self, _patch_deps) -> None:
        mock_sm, _, _ = _patch_deps
        mock_sm.audit_state.side_effect = [
            AuditResult(
                issues=[
                    AuditIssue("orphaned_display_name", "@7", fixable=True),
                    AuditIssue("stale_offset", "user 1, @9", fixable=True),
                ],
                total_bindings=1,
                live_binding_count=1,
            ),
            AuditResult(
                issues=[
                    AuditIssue("stale_offset", "user 1, @9", fixable=True),
                ],
                total_bindings=1,
                live_binding_count=1,
            ),
        ]

        query = AsyncMock()

        with patch("ccbot.handlers.sync_command.safe_edit") as mock_edit:
            await handle_sync_fix(query)
            assert "\u2705 Fixed 1 issue" in mock_edit.call_args[0][1]
