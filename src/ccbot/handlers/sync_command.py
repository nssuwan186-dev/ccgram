"""On-demand state audit and cleanup — /sync command.

Audits all state maps against live tmux windows and reports issues.
A "Fix" button runs cleanup operations and re-audits in place.

Key functions:
  - sync_command(): /sync command handler
  - handle_sync_fix(): fix button callback — run cleanup, re-audit, edit in place
  - handle_sync_dismiss(): dismiss button callback — remove keyboard
"""

import structlog

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import ContextTypes

from ..config import config
from ..session import AuditResult, session_manager
from ..tmux_manager import tmux_manager
from .callback_data import CB_SYNC_DISMISS, CB_SYNC_FIX
from .message_sender import safe_edit, safe_reply

logger = structlog.get_logger()

_CATEGORY_LABELS: dict[str, str] = {
    "ghost_binding": "ghost binding (dead window)",
    "orphaned_display_name": "orphaned display name",
    "orphaned_group_chat_id": "orphaned group chat ID",
    "stale_window_state": "stale window state",
    "stale_offset": "stale offset entry",
    "display_name_drift": "display name drift",
}


async def _run_audit() -> AuditResult:
    """Fetch live tmux state and run audit."""
    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]
    return session_manager.audit_state(live_ids, live_pairs)


def _format_report(
    audit: AuditResult, *, fixed_count: int = 0
) -> tuple[str, InlineKeyboardMarkup | None]:
    """Build report text and optional keyboard."""
    lines: list[str] = []

    if fixed_count > 0:
        issue_word = "issue" if fixed_count == 1 else "issues"
        lines.append(f"\u2705 Fixed {fixed_count} {issue_word}\n")
    else:
        lines.append("\U0001f50d State audit\n")

    # Binding summary
    if audit.total_bindings == 0:
        lines.append("\u2139 No topic bindings")
    elif audit.live_binding_count == audit.total_bindings:
        lines.append(f"\u2713 {audit.total_bindings} topics bound, all windows alive")
    else:
        dead = audit.total_bindings - audit.live_binding_count
        lines.append(
            f"\u26a0 {dead} ghost binding(s) "
            f"({audit.live_binding_count}/{audit.total_bindings} alive)"
        )

    # Group issues by category for summary
    category_counts: dict[str, int] = {}
    for issue in audit.issues:
        if issue.category == "ghost_binding":
            continue  # already shown in binding summary
        category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

    if category_counts:
        for cat, count in category_counts.items():
            label = _CATEGORY_LABELS.get(cat, cat)
            lines.append(f"\u26a0 {count} {label}")
    elif audit.total_bindings > 0:
        lines.append("\u2713 No orphaned entries")
        lines.append("\u2713 Display names in sync")

    text = "\n".join(lines)

    # Build keyboard
    fixable = audit.fixable_count
    if fixable > 0:
        issue_word = "issue" if fixable == 1 else "issues"
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        f"\U0001f527 Fix {fixable} {issue_word}",
                        callback_data=CB_SYNC_FIX,
                    ),
                    InlineKeyboardButton(
                        "\u2715 Dismiss", callback_data=CB_SYNC_DISMISS
                    ),
                ]
            ]
        )
    else:
        keyboard = None

    return text, keyboard


async def sync_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sync — audit state and show report."""
    user = update.effective_user
    if not user or not update.message:
        return

    if not config.is_user_allowed(user.id):
        await safe_reply(update.message, "You are not authorized to use this bot.")
        return

    audit = await _run_audit()
    text, keyboard = _format_report(audit)
    await safe_reply(update.message, text, reply_markup=keyboard)


async def handle_sync_fix(query: CallbackQuery) -> None:
    """Run all fix operations, re-audit, and edit message in place."""
    # Single list_windows call — reused for both audit and fix
    all_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in all_windows}
    live_pairs = [(w.window_id, w.window_name) for w in all_windows]

    # Audit before fixing to count fixable issues
    pre_audit = session_manager.audit_state(live_ids, live_pairs)

    # Run all fix operations, catch errors to still report partial results
    try:
        session_manager.sync_display_names(live_pairs)
        session_manager.prune_stale_state(live_ids)
        session_manager.prune_session_map(live_ids)
        session_manager.prune_stale_window_states(live_ids)
        # Capture state_ids AFTER prune_stale_window_states so pruned states
        # are excluded from the "known" set for offset pruning
        bound_ids: set[str] = {
            wid for _uid, _tid, wid in session_manager.iter_thread_bindings()
        }
        state_ids = set(session_manager.window_states.keys())
        session_manager.prune_stale_offsets(live_ids | bound_ids | state_ids)
    except OSError:
        logger.exception("Error during sync fix operations")

    # Re-audit and compute actual fixed count (handles partial failures)
    post_audit = await _run_audit()
    actual_fixed = pre_audit.fixable_count - post_audit.fixable_count
    text, keyboard = _format_report(post_audit, fixed_count=actual_fixed)
    await safe_edit(query, text, reply_markup=keyboard)


async def handle_sync_dismiss(query: CallbackQuery) -> None:
    """Remove keyboard from sync message."""
    original_text = getattr(query.message, "text", None) if query.message else None
    await safe_edit(query, original_text or "Dismissed", reply_markup=None)
