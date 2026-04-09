import pytest

from app.domain.session_context import SessionContext


def test_get_messages_returns_deepcopy_and_prevents_external_mutation():
    ctx = SessionContext(messages=[{"role": "user", "content": "hello"}])

    messages = ctx.get_messages()
    messages.append({"role": "assistant", "content": "polluted"})
    messages[0]["content"] = "changed-outside"

    latest = ctx.get_messages()
    assert len(latest) == 1
    assert latest[0]["content"] == "hello"


def test_add_message_rejects_invalid_role():
    ctx = SessionContext()
    with pytest.raises(ValueError):
        ctx.add_message({"role": "tool", "content": "not-allowed"})


def test_add_message_rejects_non_string_content():
    ctx = SessionContext()
    with pytest.raises(TypeError):
        ctx.add_message({"role": "user", "content": {"x": 1}})


def test_message_count_cannot_exceed_limit():
    ctx = SessionContext(max_messages=2)
    ctx.add_message({"role": "system", "content": "s"})
    ctx.add_message({"role": "user", "content": "u"})

    with pytest.raises(ValueError):
        ctx.add_message({"role": "assistant", "content": "a"})


def test_prepend_message_puts_message_at_beginning():
    ctx = SessionContext(messages=[{"role": "user", "content": "hello"}])
    ctx.prepend_message({"role": "system", "content": "sys"})

    messages = ctx.get_messages()
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_init_does_not_hold_external_list_reference():
    external_messages = [{"role": "user", "content": "hello"}]
    ctx = SessionContext(messages=external_messages)

    external_messages.append({"role": "assistant", "content": "outside"})
    external_messages[0]["content"] = "changed"

    latest = ctx.get_messages()
    assert len(latest) == 1
    assert latest[0]["content"] == "hello"


def test_returned_messages_append_illegal_data_does_not_pollute_internal_state():
    ctx = SessionContext(messages=[{"role": "user", "content": "hello"}])

    leaked = ctx.get_messages()
    leaked.append({"role": "hacker", "content": {"bad": True}})

    latest = ctx.get_messages()
    assert latest == [{"role": "user", "content": "hello"}]

    # 内部仍保持可用，且继续执行合法写入不受影响
    ctx.add_message({"role": "assistant", "content": "ok"})
    assert len(ctx.get_messages()) == 2


def test_check_rep_detects_tampered_internal_state():
    ctx = SessionContext(messages=[{"role": "user", "content": "hello"}])
    ctx._messages.append({"role": "tool", "content": "bad"})  # 人为破坏内部状态

    with pytest.raises(AssertionError):
        ctx._check_rep()
