"""Periodic tasks for the polling subsystem.

Time-gated tasks that run on different intervals within the poll loop:
topic existence probing, state pruning, autoclose timers, unbound window TTL,
message broker delivery, and mailbox sweep.

Key components:
  - run_periodic_tasks: time-gated broker, sweep, and topic check
  - run_lifecycle_tasks: per-tick autoclose and unbound window management
  - run_broker_cycle: message broker delivery (also called from hook_events)
"""

import time
from typing import TYPE_CHECKING

import structlog
from telegram import Bot
from telegram.error import BadRequest, TelegramError

from ..config import config
from ..session import session_manager
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from ..utils import log_throttle_sweep, log_throttled
from ..window_resolver import is_foreign_window
from .cleanup import clear_topic_state
from .msg_broker import BROKER_CYCLE_INTERVAL, SWEEP_INTERVAL
from .polling_strategies import (
    clear_window_poll_state,
    lifecycle_strategy,
    terminal_strategy,
)

if TYPE_CHECKING:
    from ..tmux_manager import TmuxWindow

logger = structlog.get_logger()

# ── Timing constants ──────────────────────────────────────────────────────

TOPIC_CHECK_INTERVAL = 60.0  # seconds


# ── Broker integration ────────────────────────────────────────────────────


async def run_broker_cycle(
    bot: Bot | None = None,
    idle_windows: frozenset[str] = frozenset(),
) -> None:
    """Run one broker delivery cycle (called from poll loop and hook_events)."""
    from ..mailbox import Mailbox

    from .msg_broker import broker_delivery_cycle

    mailbox = Mailbox(config.mailbox_dir)
    await broker_delivery_cycle(
        mailbox=mailbox,
        tmux_mgr=tmux_manager,
        window_states=session_manager.window_states,
        tmux_session=config.tmux_session_name,
        msg_rate_limit=config.msg_rate_limit,
        bot=bot,
        idle_windows=idle_windows,
    )


def _run_mailbox_sweep() -> None:
    """Run periodic mailbox sweep."""
    from ..mailbox import Mailbox

    mailbox = Mailbox(config.mailbox_dir)
    removed = mailbox.sweep()
    if removed:
        logger.debug("Mailbox sweep removed %d messages", removed)


# ── Autoclose timer management ────────────────────────────────────────────


async def _check_autoclose_timers(bot: Bot) -> None:
    """Close topics whose done/dead timers have expired."""
    all_topics = lifecycle_strategy.iter_autoclose_timers()
    if not all_topics:
        return

    now = time.monotonic()
    expired: list[tuple[int, int]] = []
    for user_id, thread_id, ts in all_topics:
        if ts.autoclose is None:
            continue
        state, entered_at = ts.autoclose
        if state == "done":
            timeout = config.autoclose_done_minutes * 60
        elif state == "dead":
            timeout = config.autoclose_dead_minutes * 60
        else:
            continue
        if timeout > 0 and now - entered_at >= timeout:
            expired.append((user_id, thread_id))

    for user_id, thread_id in expired:
        await _close_expired_topic(bot, user_id, thread_id)


async def _close_expired_topic(bot: Bot, user_id: int, thread_id: int) -> None:
    """Attempt to close/delete an expired topic and clean up state."""
    chat_id = thread_router.resolve_chat_id(user_id, thread_id)
    window_id = thread_router.get_window_for_thread(user_id, thread_id)
    removed = False
    try:
        await bot.delete_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
        removed = True
    except TelegramError:
        try:
            await bot.close_forum_topic(chat_id=chat_id, message_thread_id=thread_id)
            removed = True
        except TelegramError as e:
            logger.debug("autoclose_failed", thread_id=thread_id, error=str(e))
    if removed:
        lifecycle_strategy.clear_autoclose_timer(user_id, thread_id)
        logger.info(
            "auto_removed_topic", chat_id=chat_id, thread_id=thread_id, user_id=user_id
        )
        await clear_topic_state(
            user_id,
            thread_id,
            bot=bot,
            window_id=window_id,
            window_dead=True,
        )
        thread_router.unbind_thread(user_id, thread_id)


# ── Unbound window TTL ────────────────────────────────────────────────────


async def _check_unbound_window_ttl(
    live_windows: "list[TmuxWindow] | None" = None,
) -> None:
    """Kill unbound tmux windows whose TTL has expired."""
    timeout = config.autoclose_done_minutes * 60
    if timeout <= 0:
        return

    bound_ids: set[str] = set()
    for _, _, wid in thread_router.iter_thread_bindings():
        bound_ids.add(wid)

    if live_windows is None:
        live_windows = await tmux_manager.list_windows()
    live_ids = {w.window_id for w in live_windows}

    terminal_strategy.clear_unbound_timers(bound_ids, live_ids)

    now = time.monotonic()
    for w in live_windows:
        if w.window_id not in bound_ids and not is_foreign_window(w.window_id):
            ws = terminal_strategy.get_state(w.window_id)
            if ws.unbound_timer is None:
                terminal_strategy.set_unbound_timer(w.window_id, now)

    await _kill_expired_unbound(now, timeout)
    _prune_orphaned_poll_state(live_ids, bound_ids)


async def _kill_expired_unbound(now: float, timeout: float) -> None:
    """Find and kill unbound windows past their TTL."""
    expired = terminal_strategy.get_expired_unbound(now, timeout)
    for wid in expired:
        await tmux_manager.kill_window(wid)

        from ..topic_state_registry import topic_state

        topic_state.clear_window(wid)
        qualified_id = (
            wid if is_foreign_window(wid) else f"{config.tmux_session_name}:{wid}"
        )
        topic_state.clear_qualified(qualified_id)
        logger.info("auto_killed_unbound_window", window_id=wid)


def _prune_orphaned_poll_state(live_ids: set[str], bound_ids: set[str]) -> None:
    """Remove poll state for windows that are neither live nor bound."""
    for wid in terminal_strategy.get_orphaned_window_ids(live_ids, bound_ids):
        clear_window_poll_state(wid)


# ── Display name sync / state pruning ─────────────────────────────────────


async def _prune_stale_state(live_windows: "list[TmuxWindow]") -> None:
    """Sync display names and prune orphaned state entries."""
    live_ids = {w.window_id for w in live_windows}
    live_pairs = [(w.window_id, w.window_name) for w in live_windows]
    session_manager.sync_display_names(live_pairs)
    session_manager.prune_stale_state(live_ids)


# ── Topic existence probing ───────────────────────────────────────────────


async def _probe_topic_existence(bot: Bot) -> None:
    """Probe all bound topics via Telegram API; detect deleted topics."""
    for user_id, thread_id, wid in list(thread_router.iter_thread_bindings()):
        if lifecycle_strategy.should_skip_probe(wid):
            continue
        try:
            await bot.unpin_all_forum_topic_messages(
                chat_id=thread_router.resolve_chat_id(user_id, thread_id),
                message_thread_id=thread_id,
            )
            terminal_strategy.reset_probe_failures(wid)
        except TelegramError as e:
            if isinstance(e, BadRequest) and (
                "Topic_id_invalid" in e.message
                or "thread not found" in e.message.lower()
            ):
                w = await tmux_manager.find_window_by_id(wid)
                if w:
                    await tmux_manager.kill_window(w.window_id)
                terminal_strategy.reset_probe_failures(wid)
                await clear_topic_state(user_id, thread_id, bot, window_id=wid)
                thread_router.unbind_thread(user_id, thread_id)
                logger.info(
                    "Topic deleted: killed window_id '%s' and "
                    "unbound thread %d for user %d",
                    wid,
                    thread_id,
                    user_id,
                )
            else:
                lifecycle_strategy.record_probe_failure(wid)
                if not lifecycle_strategy.should_skip_probe(wid):
                    log_throttled(
                        logger,
                        f"topic-probe:{wid}",
                        "Topic probe error for %s: %s",
                        wid,
                        e,
                    )


# ── Orchestration ──────────────────────────────────────────────────────────


async def run_periodic_tasks(
    bot: Bot,
    all_windows: list["TmuxWindow"],
    timers: dict[str, float],
) -> None:
    """Run time-gated periodic tasks (topic check, broker, sweep)."""
    now = time.monotonic()
    if now - timers["topic_check"] >= TOPIC_CHECK_INTERVAL:
        timers["topic_check"] = now
        await _prune_stale_state(all_windows)
        await _probe_topic_existence(bot)
        log_throttle_sweep()

    if now - timers["broker"] >= BROKER_CYCLE_INTERVAL:
        timers["broker"] = now
        await run_broker_cycle(bot)

    if now - timers["sweep"] >= SWEEP_INTERVAL:
        timers["sweep"] = now
        _run_mailbox_sweep()


async def run_lifecycle_tasks(bot: Bot, all_windows: list["TmuxWindow"]) -> None:
    """Run per-tick lifecycle tasks (autoclose timers, unbound window TTL)."""
    await _check_autoclose_timers(bot)
    await _check_unbound_window_ttl(all_windows)
