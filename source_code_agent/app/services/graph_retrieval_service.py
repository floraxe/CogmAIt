import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.utils.llm_knowledge_extractor import LLMKnowledgeExtractor
from app.utils.neo4j_utils import get_neo4j_service
from app.utils.config import get_neo4j_config
from app.utils.model import execute_model_inference
from app.utils.graph import get_graph
from app.services import chat_events as ev

logger = logging.getLogger(__name__)


class GraphRetrievalService:
    async def stream_graph_events(
        self,
        db: Session,
        agent: Any,
        user_message: str,
        model_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not getattr(agent, "graphs", None):
            return

        graph_list: Dict[str, Any] = {"nodes": [], "links": []}
        yield ev.graph_search_start().to_dict()

        try:
            neo4j_config = get_neo4j_config()
            yield ev.graph_connecting().to_dict()

            neo4j_service = get_neo4j_service(
                uri=neo4j_config.get("uri"),
                username=neo4j_config.get("username"),
                password=neo4j_config.get("password"),
                database=neo4j_config.get("database"),
                force_new=True,
            )
            if not neo4j_service or not neo4j_service.driver or not neo4j_service.is_connected():
                yield ev.graph_connection_error().to_dict()
                return

            yield ev.graph_connected().to_dict()

            extractor = LLMKnowledgeExtractor(db=db)
            for gb in agent.graphs:
                graph = get_graph(db, gb.id)
                if not graph or not graph.neo4j_subgraph:
                    continue

                schema = None
                try:
                    from app.utils.graph import get_graph_schema
                    schema = get_graph_schema(db, graph.id)
                    yield ev.graph_schema_ready().to_dict()
                except Exception:
                    logger.exception("获取图谱 Schema 失败")
                    yield ev.graph_schema_error().to_dict()

                yield ev.graph_analysis().to_dict()

                extraction_prompt = (
                    f'请分析用户问题并给出Neo4j Cypher查询，仅返回JSON:\n'
                    f'{{"cypher": "MATCH (n) RETURN n LIMIT 15"}}\n'
                    f'用户问题: "{user_message}"\n'
                    f'图谱Schema: {json.dumps(schema, ensure_ascii=False) if schema else "未定义schema"}\n'
                    f'子图名: {graph.neo4j_subgraph}'
                )

                extraction_result = await execute_model_inference(
                    db,
                    model_id,
                    {
                        "messages": [
                            {"role": "system", "content": "你是一个生成Neo4j Cypher查询的助手。"},
                            {"role": "user", "content": extraction_prompt},
                        ],
                        "model_type": "chat",
                    },
                )

                yield ev.graph_query_generated().to_dict()

                cypher_query = self._extract_cypher(extraction_result, user_message, graph.neo4j_subgraph)
                if not cypher_query:
                    yield ev.graph_query_invalid().to_dict()
                    continue

                yield ev.graph_search_running().to_dict()
                try:
                    with neo4j_service.driver.session(database=neo4j_service.database) as session:
                        records = list(session.run(cypher_query))
                    if not records:
                        yield ev.graph_search_empty().to_dict()
                        continue

                    graph_list = self._build_graph_list(records)
                    yield ev.graph_search_complete(graph_list).to_dict()
                except Exception:
                    logger.exception("执行 Cypher 查询失败")
                    yield ev.graph_search_error().to_dict()

            yield ev.graph_search_complete(graph_list).to_dict()
        except Exception:
            logger.exception("图谱检索整体失败")
            yield ev.graph_search_error().to_dict()

    async def run(self, db: Session, agent: Any, user_message: str, model_id: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        async for item in self.stream_graph_events(db=db, agent=agent, user_message=user_message, model_id=model_id):
            events.append(item)
        return events

    @staticmethod
    def _extract_cypher(extraction_result: Any, user_message: str, subgraph_name: str) -> Optional[str]:
        subgraph_id = (subgraph_name or "").lower().replace(" ", "_").replace("-", "_")
        message_content = ""
        if isinstance(extraction_result, dict):
            choices = extraction_result.get("choices") or []
            if choices:
                message_content = choices[0].get("message", {}).get("content", "")
        elif isinstance(extraction_result, str):
            message_content = extraction_result

        if message_content:
            code_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', message_content)
            try:
                payload = json.loads(code_match.group(1) if code_match else message_content)
                if isinstance(payload, dict) and payload.get("cypher"):
                    return GraphRetrievalService._normalize_cypher(payload["cypher"], subgraph_id, user_message)
            except Exception:
                pass

            for pattern in [
                r'```cypher\s*(MATCH[\s\S]+?)\s*```',
                r'```\s*(MATCH[\s\S]+?)\s*```',
                r'(MATCH\s*\([^)]+\)[\s\S]+?RETURN[^;]+)',
            ]:
                matches = re.findall(pattern, message_content, re.IGNORECASE)
                if matches:
                    return GraphRetrievalService._normalize_cypher(matches[0].strip(), subgraph_id, user_message)

        fallback_entity = user_message.strip().replace("'", "")
        return f"MATCH (n) WHERE n.graph_id = '{subgraph_id}' AND n.name CONTAINS '{fallback_entity}' RETURN n LIMIT 20"

    @staticmethod
    def _normalize_cypher(cypher_query: str, subgraph_id: str, user_message: str) -> str:
        cypher_query = re.sub(r"MATCH\s*\(\w+:\w+\)", "MATCH (n)", cypher_query)
        cypher_query = cypher_query.replace("...", "").strip().replace('"', "'").replace("''", "'")
        if "WHERE" in cypher_query.upper():
            where_pos = cypher_query.upper().find("WHERE")
            return_pos = cypher_query.upper().find("RETURN", where_pos)
            where_part = cypher_query[where_pos:return_pos] if return_pos != -1 else cypher_query[where_pos:]
            if "graph_id" not in where_part.lower():
                if return_pos != -1:
                    new_where = f"WHERE n.graph_id = '{subgraph_id}' AND " + where_part[5:].strip()
                    cypher_query = cypher_query[:where_pos] + new_where + cypher_query[return_pos:]
                else:
                    cypher_query = cypher_query.replace("WHERE", f"WHERE n.graph_id = '{subgraph_id}' AND ")
        else:
            return_pos = cypher_query.upper().find("RETURN")
            if return_pos > 0:
                cypher_query = (
                    cypher_query[:return_pos]
                    + f" WHERE n.graph_id = '{subgraph_id}' "
                    + cypher_query[return_pos:]
                )
            else:
                entity = user_message.strip().replace("'", "")
                cypher_query = (
                    f"MATCH (n) WHERE n.graph_id = '{subgraph_id}' "
                    f"AND n.name CONTAINS '{entity}' RETURN n LIMIT 20"
                )
        return cypher_query

    @staticmethod
    def _build_graph_list(records: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        nodes: Dict[str, Dict[str, Any]] = {}
        links: Dict[str, Dict[str, Any]] = {}
        for record in records:
            for _, value in record.items():
                if value is None:
                    continue
                if hasattr(value, "id") and hasattr(value, "labels"):
                    node_id = str(value.id)
                    props = dict(value.properties) if hasattr(value, "properties") else {}
                    node_name = props.get("name") or props.get("title") or f"节点{node_id}"
                    node_type = list(value.labels)[0] if getattr(value, "labels", None) else "Entity"
                    nodes[node_id] = {
                        "id": node_id,
                        "name": str(node_name),
                        "symbolSize": 50,
                        "category": str(node_type),
                        "properties": props,
                    }
                elif hasattr(value, "type") and hasattr(value, "start_node") and hasattr(value, "end_node"):
                    start_id = str(value.start_node.id)
                    end_id = str(value.end_node.id)
                    rel_type = str(value.type)
                    link_id = f"{start_id}_{rel_type}_{end_id}"
                    links[link_id] = {
                        "source": start_id,
                        "target": end_id,
                        "value": rel_type,
                        "properties": dict(value.properties) if hasattr(value, "properties") else {},
                    }
        return {"nodes": list(nodes.values()), "links": list(links.values())}
