from typing import Any, Dict, List

AGENT_TYPES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "问答助手",
        "value": "qa_bot",
        "description": "基于知识库的问答智能体，专注于精确回答",
    },
    {
        "id": 2,
        "name": "对话助手",
        "value": "chat",
        "description": "通用对话智能体，适合开放式对话场景",
    },
    {
        "id": 3,
        "name": "知识库助手",
        "value": "knowledge_assistant",
        "description": "基于知识库的智能体，提供详细的知识解答",
    },
    {
        "id": 4,
        "name": "图谱问答",
        "value": "graph_qa",
        "description": "基于知识图谱的智能体，善于处理结构化信息",
    },
    {
        "id": 5,
        "name": "混合增强",
        "value": "rag",
        "description": "同时使用知识库和知识图谱的高级智能体",
    },
]
