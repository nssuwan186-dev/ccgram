from unittest.mock import MagicMock

import pytest

from ccgram.handlers.topic_state_registry import TopicStateRegistry


@pytest.fixture
def registry():
    return TopicStateRegistry()


class TestRegister:
    def test_register_topic_scope(self, registry: TopicStateRegistry):
        fn = MagicMock()
        registry.register("topic")(fn)
        assert fn in registry._cleanups["topic"]

    def test_register_window_scope(self, registry: TopicStateRegistry):
        fn = MagicMock()
        registry.register("window")(fn)
        assert fn in registry._cleanups["window"]

    def test_register_qualified_scope(self, registry: TopicStateRegistry):
        fn = MagicMock()
        registry.register("qualified")(fn)
        assert fn in registry._cleanups["qualified"]

    def test_register_chat_scope(self, registry: TopicStateRegistry):
        fn = MagicMock()
        registry.register("chat")(fn)
        assert fn in registry._cleanups["chat"]

    def test_register_invalid_scope_raises(self, registry: TopicStateRegistry):
        with pytest.raises(ValueError, match="Unknown cleanup scope"):
            registry.register("nonexistent_scope")

    def test_dedup_prevents_double_registration(self, registry: TopicStateRegistry):
        fn = MagicMock()
        registry.register("topic")(fn)
        registry.register("topic")(fn)
        assert registry._cleanups["topic"].count(fn) == 1


class TestClearTopic:
    def test_calls_all_topic_scoped(self, registry: TopicStateRegistry):
        mocks = [MagicMock() for _ in range(3)]
        for m in mocks:
            registry.register("topic")(m)
        registry.clear_topic(user_id=1, thread_id=42)
        for m in mocks:
            m.assert_called_once_with(1, 42)

    def test_no_registered_functions(self, registry: TopicStateRegistry):
        registry.clear_topic(user_id=1, thread_id=42)

    def test_failing_cleanup_doesnt_block_others(self, registry: TopicStateRegistry):
        first = MagicMock()
        failing = MagicMock(side_effect=RuntimeError("boom"))
        third = MagicMock()
        registry.register("topic")(first)
        registry.register("topic")(failing)
        registry.register("topic")(third)

        registry.clear_topic(user_id=1, thread_id=42)

        first.assert_called_once()
        third.assert_called_once()


class TestClearWindow:
    def test_calls_all_window_scoped(self, registry: TopicStateRegistry):
        mocks = [MagicMock() for _ in range(3)]
        for m in mocks:
            registry.register("window")(m)
        registry.clear_window(window_id="@0")
        for m in mocks:
            m.assert_called_once_with("@0")


class TestClearQualified:
    def test_calls_all_qualified_scoped(self, registry: TopicStateRegistry):
        mocks = [MagicMock() for _ in range(2)]
        for m in mocks:
            registry.register("qualified")(m)
        registry.clear_qualified(qualified_id="ccgram:@0")
        for m in mocks:
            m.assert_called_once_with("ccgram:@0")


class TestClearChat:
    def test_calls_all_chat_scoped(self, registry: TopicStateRegistry):
        mocks = [MagicMock() for _ in range(2)]
        for m in mocks:
            registry.register("chat")(m)
        registry.clear_chat(chat_id=99, thread_id=42)
        for m in mocks:
            m.assert_called_once_with(99, 42)


class TestClearAll:
    def test_dispatches_all_scopes(self, registry: TopicStateRegistry):
        topic_fn = MagicMock()
        window_fn = MagicMock()
        qualified_fn = MagicMock()
        chat_fn = MagicMock()
        registry.register("topic")(topic_fn)
        registry.register("window")(window_fn)
        registry.register("qualified")(qualified_fn)
        registry.register("chat")(chat_fn)

        registry.clear_all(
            user_id=1,
            thread_id=42,
            window_id="@0",
            qualified_id="ccgram:@0",
            chat_id=99,
        )

        topic_fn.assert_called_once_with(1, 42)
        window_fn.assert_called_once_with("@0")
        qualified_fn.assert_called_once_with("ccgram:@0")
        chat_fn.assert_called_once_with(99, 42)

    def test_skips_missing_window_id(self, registry: TopicStateRegistry):
        topic_fn = MagicMock()
        window_fn = MagicMock()
        registry.register("topic")(topic_fn)
        registry.register("window")(window_fn)

        registry.clear_all(user_id=1, thread_id=42)

        topic_fn.assert_called_once_with(1, 42)
        window_fn.assert_not_called()

    def test_skips_missing_qualified_id(self, registry: TopicStateRegistry):
        qualified_fn = MagicMock()
        registry.register("qualified")(qualified_fn)

        registry.clear_all(user_id=1, thread_id=42, window_id="@0")

        qualified_fn.assert_not_called()

    def test_skips_missing_chat_id(self, registry: TopicStateRegistry):
        chat_fn = MagicMock()
        registry.register("chat")(chat_fn)

        registry.clear_all(user_id=1, thread_id=42)

        chat_fn.assert_not_called()

    def test_ordering_topic_before_window(self, registry: TopicStateRegistry):
        order: list[str] = []
        topic_fn = MagicMock(side_effect=lambda *_: order.append("topic"))
        window_fn = MagicMock(side_effect=lambda *_: order.append("window"))
        registry.register("topic")(topic_fn)
        registry.register("window")(window_fn)

        registry.clear_all(user_id=1, thread_id=42, window_id="@0")

        assert order.index("topic") < order.index("window")

    def test_full_lifecycle_5_handlers(self, registry: TopicStateRegistry):
        fns = [MagicMock() for _ in range(5)]
        registry.register("topic")(fns[0])
        registry.register("topic")(fns[1])
        registry.register("window")(fns[2])
        registry.register("qualified")(fns[3])
        registry.register("chat")(fns[4])

        registry.clear_all(
            user_id=1,
            thread_id=42,
            window_id="@0",
            qualified_id="ccgram:@0",
            chat_id=99,
        )

        for fn in fns:
            fn.assert_called_once()
