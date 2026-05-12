"""
兼容入口：智能体相关持久化实现位于 app.services.agent_service。

新代码请直接 ``from app.services import agent_service``。
"""
from app.services.agent_service import (  # noqa: F401
    count_agents,
    count_chat_history,
    create_agent,
    create_chat_history,
    delete_agent,
    delete_api_key,
    delete_share_token,
    execute_model_inference,
    generate_api_key,
    generate_random_token,
    generate_share_token,
    get_agent,
    get_agent_api_keys,
    get_agent_by_api_key,
    get_agent_by_share_token,
    get_agent_share_tokens,
    get_agents,
    get_chat_history,
    get_chat_sessions,
    get_graph,
    get_model,
    toggle_api_status,
    toggle_share_status,
    update_agent,
)
