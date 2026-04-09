from app.domain.memory import MemoryManager


def test_memory_manager_assembles_messages_in_order():
    memory = MemoryManager()
    memory.add_system_prompt("system")
    memory.add_history([{"role": "user", "content": "hello"}])
    memory.add_tool_result("tool-result")

    messages = memory.messages()
    assert [msg["role"] for msg in messages] == ["system", "user", "system"]
    assert messages[1]["content"] == "hello"


def test_memory_manager_prepend_context():
    memory = MemoryManager()
    memory.add_system_prompt("base")
    memory.prepend_context("high-priority")

    messages = memory.messages()
    assert messages[0]["content"] == "high-priority"
    assert messages[1]["content"] == "base"
