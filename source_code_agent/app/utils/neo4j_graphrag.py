"""
Neo4j GraphRAG工具模块 - 用于知识图谱构建
"""

import os
import json
import logging
import tempfile
import uuid
import random
from typing import Dict, Any, List, Optional, Tuple

from neo4j import GraphDatabase, Driver

# 配置日志
logger = logging.getLogger(__name__)

class Neo4jGraphRAG:
    """Neo4j GraphRAG工具类，用于处理知识图谱构建过程"""
    
    def __init__(
        self, 
        uri: str = None, 
        username: str = None, 
        password: str = None,
        database: str = "neo4j",
        driver: Driver = None
    ):
        """
        初始化Neo4j GraphRAG工具
        
        Args:
            uri: Neo4j数据库URI
            username: 用户名
            password: 密码
            database: 默认数据库名称
            driver: 可选，直接传入neo4j驱动实例
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        if driver:
            self.driver = driver
        elif uri and username and password:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
        else:
            self.driver = None
            logger.warning("Neo4j driver未初始化，请设置正确的连接参数")
    
    def close(self):
        """关闭驱动连接"""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def extract_knowledge_from_text(
        self,
        graph_id: str,
        text: str,
        schema: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        embedder_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从文本中提取知识并构建知识图谱
        
        Args:
            graph_id: 图谱ID
            text: 文本内容
            schema: 图谱Schema定义
            llm_config: LLM配置
            embedder_config: 向量化配置
        
        Returns:
            Dict: 提取结果
        """
        if not self.driver:
            logger.error("Neo4j driver未初始化")
            return {"success": False, "error": "Neo4j driver未初始化"}
        
        # 检查是否使用LLM进行知识抽取
        use_llm = False
        if llm_config and "use_llm" in llm_config:
            use_llm = llm_config.get("use_llm", False)
        
        # 如果配置了使用LLM进行知识抽取，调用LLMKnowledgeExtractor
        if use_llm:
            try:
                # 导入LLM知识抽取器
                from app.utils.llm_knowledge_extractor import LLMKnowledgeExtractor
                
                # 如果llm_config中包含db_session，则使用它
                db_session = llm_config.get("db_session")
                if not db_session:
                    from app.db.session import db_session as default_db
                    db_session = default_db()
                
                # 创建LLM知识抽取器
                extractor = LLMKnowledgeExtractor(
                    db=db_session,
                    neo4j_service=None  # 会自动创建Neo4j服务
                )
                
                # 获取图谱信息
                from app.models.graph import Graph
                graph_obj = db_session.query(Graph).filter(Graph.id == graph_id).first()
                
                if not graph_obj:
                    logger.error(f"无法找到ID为{graph_id}的图谱")
                    return {"success": False, "error": f"无法找到图谱: {graph_id}"}
                
                # 调用LLM知识抽取器进行异步抽取
                import asyncio
                result = asyncio.run(extractor.extract_knowledge(
                    graph=graph_obj,
                    text=text,
                    schema=schema
                ))
                
                # 记录结果
                logger.info(f"LLM知识抽取完成: 实体数={result.get('entityCount', 0)}, 关系数={result.get('relationCount', 0)}")
                
                return result
            except Exception as e:
                logger.error(f"LLM知识抽取失败: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    "success": False,
                    "error": f"LLM知识抽取失败: {str(e)}"
                }
        
        # 如果不使用LLM，使用原本的基于规则的抽取方法
        try:
            # 确保graph_id是小写且无特殊字符的格式
            subgraph_id = graph_id.lower().replace(" ", "_").replace("-", "_")
            logger.info(f"开始从文本中提取知识，子图ID: {subgraph_id}")
            
            # 准备schema信息
            entity_types = []
            relation_types = []
            
            if schema and "entityTypes" in schema:
                entity_types = schema["entityTypes"]
                logger.info(f"从schema中提取到 {len(entity_types)} 个实体类型")
            
            if schema and "relationTypes" in schema:
                relation_types = schema["relationTypes"]
                logger.info(f"从schema中提取到 {len(relation_types)} 个关系类型")
            
            # 使用简单的文本分析方法从文本中提取实体和关系
            # 这里简化处理，仅分析文本中与schema定义的实体和关系匹配的部分
            
            # 创建文档节点
            doc_id = str(uuid.uuid4())
            with self.driver.session(database=self.database) as session:
                session.run("""
                CREATE (d:Document {
                    id: $doc_id,
                    graph_id: $graph_id,
                    name: 'Document',
                    content: $content,
                    created_at: datetime()
                })
                """, doc_id=doc_id, graph_id=subgraph_id, content=text[:1000])
            
            # 统计创建的实体和关系数量
            created_entities = 0
            created_relations = 0
            
            # 基于文本分析提取实体
            # 分析文本，查找与schema中定义的实体和关系匹配的部分
            
            text_paragraphs = [p for p in text.split('\n') if p.strip()]
            text_sentences = []
            for para in text_paragraphs:
                text_sentences.extend([s.strip() for s in para.split('.') if s.strip()])
            
            # 提取的实体
            entities = []
            
            with self.driver.session(database=self.database) as session:
                # 检查是否有实体类型定义
                if entity_types:
                    for entity_type in entity_types:
                        entity_name = entity_type.get("name", "未知实体")
                        
                        # 在文本中查找与实体类型相关的内容
                        # 这里简化为查找包含实体类型名称的句子
                        matched_sentences = [
                            s for s in text_sentences 
                            if entity_name.lower() in s.lower()
                        ]
                        
                        # 为每个匹配创建实体
                        for i, sentence in enumerate(matched_sentences[:5]):  # 限制每种类型最多5个实体
                            # 生成唯一ID
                            entity_id = str(uuid.uuid4())
                            
                            # 提取实体名称 - 这里简化为使用句子的前30个字符
                            name_part = sentence[:30].strip()
                            if not name_part:
                                name_part = f"{entity_name}_{i+1}"
                            
                            # 创建实体节点
                            session.run(f"""
                            CREATE (e:{entity_name} {{
                                id: $entity_id,
                                graph_id: $graph_id,
                                name: $name,
                                description: $description,
                                created_at: datetime(),
                                source: 'extracted'
                            }})
                            """, 
                                entity_id=entity_id, 
                                graph_id=subgraph_id, 
                                name=name_part,
                                description=sentence
                            )
                            
                            # 将实体与文档关联
                            session.run(f"""
                            MATCH (d:Document {{id: $doc_id, graph_id: $graph_id}}),
                                  (e:{entity_name} {{id: $entity_id, graph_id: $graph_id}})
                            CREATE (d)-[r:HAS_ENTITY {{
                                graph_id: $graph_id,
                                created_at: datetime()
                            }}]->(e)
                            """, doc_id=doc_id, entity_id=entity_id, graph_id=subgraph_id)
                            
                            entities.append({
                                "id": entity_id,
                                "type": entity_name,
                                "name": name_part
                            })
                            
                            created_entities += 1
                else:
                    # 如果没有实体类型定义，则使用简单的命名实体识别逻辑
                    # 这里简化处理，仅提取可能的人名、组织名和地点名
                    # 真实场景应使用适当的NLP库
                    
                    # 简单示例：将长句分割，每个分割作为一个潜在实体
                    potential_entities = []
                    for sentence in text_sentences:
                        if len(sentence) > 10:  # 只处理较长的句子
                            words = sentence.split()
                            if len(words) > 3:
                                # 可能的实体名词组
                                for i in range(len(words) - 2):
                                    if len(words[i]) > 1 and words[i][0].isupper():
                                        potential_entities.append(" ".join(words[i:i+3]))
                    
                    # 创建通用实体
                    for i, entity_text in enumerate(potential_entities[:10]):  # 限制最多10个通用实体
                        entity_id = str(uuid.uuid4())
                        
                        session.run("""
                        CREATE (e:Entity {
                            id: $entity_id,
                            graph_id: $graph_id,
                            name: $name,
                            description: $description,
                            created_at: datetime(),
                            source: 'extracted'
                        })
                        """, 
                            entity_id=entity_id, 
                            graph_id=subgraph_id, 
                            name=entity_text[:30],
                            description=entity_text
                        )
                        
                        # 将实体与文档关联
                        session.run("""
                        MATCH (d:Document {id: $doc_id, graph_id: $graph_id}),
                              (e:Entity {id: $entity_id, graph_id: $graph_id})
                        CREATE (d)-[r:HAS_ENTITY {
                            graph_id: $graph_id,
                            created_at: datetime()
                        }]->(e)
                        """, doc_id=doc_id, entity_id=entity_id, graph_id=subgraph_id)
                        
                        entities.append({
                            "id": entity_id,
                            "type": "Entity",
                            "name": entity_text[:30]
                        })
                        
                        created_entities += 1
                
                # 如果实体数量至少为2，创建关系
                if created_entities >= 2:
                    # 获取所有已创建的实体
                    entities_result = session.run("""
                    MATCH (e)
                    WHERE e.graph_id = $graph_id AND NOT e:Document AND NOT e:GraphMetadata
                    RETURN id(e) AS id, labels(e) AS labels, e.name AS name, e.id AS entity_id
                    """, graph_id=subgraph_id)
                    
                    entity_nodes = list(entities_result)
                    
                    # 提取关系
                    if relation_types and len(entity_nodes) >= 2:
                        for relation in relation_types:
                            rel_type = relation.get("name", "RELATED_TO")
                            source_type = relation.get("sourceType")
                            target_type = relation.get("targetType")
                            
                            # 查找源实体和目标实体
                            source_entities = []
                            target_entities = []
                            
                            for node in entity_nodes:
                                entity_label = node["labels"][0] if node["labels"] else None
                                
                                # 找到源实体类型的节点
                                source_entity_type = next((e for e in entity_types if str(e.get("id")) == str(source_type)), None)
                                if source_entity_type and entity_label == source_entity_type.get("name"):
                                    source_entities.append(node)
                                
                                # 找到目标实体类型的节点
                                target_entity_type = next((e for e in entity_types if str(e.get("id")) == str(target_type)), None)
                                if target_entity_type and entity_label == target_entity_type.get("name"):
                                    target_entities.append(node)
                            
                            # 创建关系
                            relations_created = 0
                            for source in source_entities:
                                for target in target_entities:
                                    if relations_created >= 3:  # 限制每种关系类型最多创建3个关系
                                        break
                                    
                                    # 创建关系
                                    session.run(f"""
                                    MATCH (a) WHERE id(a) = $source_id
                                    MATCH (b) WHERE id(b) = $target_id
                                    CREATE (a)-[r:{rel_type} {{
                                        graph_id: $graph_id,
                                        description: '从文本抽取的关系',
                                        created_at: datetime(),
                                        source: 'extracted'
                                    }}]->(b)
                                    """, 
                                        source_id=source["id"], 
                                        target_id=target["id"],
                                        graph_id=subgraph_id
                                    )
                                    
                                    created_relations += 1
                                    relations_created += 1
                                    
                                if relations_created >= 3:
                                    break
                    
                    # 如果没有定义关系类型，或者使用关系类型创建的关系数量为0
                    # 则尝试基于实体共现在文本中的情况创建关系
                    if created_relations == 0 and len(entity_nodes) >= 2:
                        # 选择部分实体创建关系
                        for i in range(min(len(entity_nodes) - 1, 5)):
                            source_node = entity_nodes[i]
                            target_node = entity_nodes[(i + 1) % len(entity_nodes)]
                            
                            # 创建通用关系
                            session.run("""
                            MATCH (a) WHERE id(a) = $source_id
                            MATCH (b) WHERE id(b) = $target_id
                            CREATE (a)-[r:RELATED_TO {
                                graph_id: $graph_id,
                                description: '基于文本共现创建的关系',
                                created_at: datetime(),
                                source: 'extracted'
                            }]->(b)
                            """, 
                                source_id=source_node["id"], 
                                target_id=target_node["id"],
                                graph_id=subgraph_id
                            )
                            
                            created_relations += 1
            
            logger.info(f"知识抽取完成: 创建了 {created_entities} 个实体和 {created_relations} 个关系")
            
            # 返回结果
            return {
                "success": True,
                "message": "知识图谱构建完成",
                "entityCount": created_entities,
                "relationCount": created_relations
            }
        
        except Exception as e:
            logger.error(f"知识提取失败: {str(e)}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _convert_entity_type(self, entity_type: Dict[str, Any]) -> Dict[str, Any]:
        """转换实体类型为neo4j-graphrag格式"""
        if isinstance(entity_type, str):
            return {"label": entity_type}
        
        result = {"label": entity_type.get("name", "Unknown")}
        
        if "description" in entity_type:
            result["description"] = entity_type["description"]
        
        # 处理属性
        if "properties" in entity_type:
            props = []
            for prop in entity_type.get("properties", []):
                if isinstance(prop, dict):
                    props.append({
                        "name": prop.get("name"),
                        "type": prop.get("type", "STRING")
                    })
            if props:
                result["properties"] = props
        
        return result
    
    def _convert_relation_type(self, relation_type: Dict[str, Any]) -> Dict[str, Any]:
        """转换关系类型为neo4j-graphrag格式"""
        if isinstance(relation_type, str):
            return {"label": relation_type}
        
        result = {"label": relation_type.get("name", "RELATED_TO")}
        
        if "description" in relation_type:
            result["description"] = relation_type["description"]
        
        # 处理属性
        if "properties" in relation_type:
            props = []
            for prop in relation_type.get("properties", []):
                if isinstance(prop, dict):
                    props.append({
                        "name": prop.get("name"),
                        "type": prop.get("type", "STRING")
                    })
            if props:
                result["properties"] = props
        
        return result


# 单例模式全局实例
_neo4j_graphrag_instance = None

def get_neo4j_graphrag(
    uri: str = None, 
    username: str = None, 
    password: str = None,
    database: str = "neo4j",
    force_new: bool = False
) -> Neo4jGraphRAG:
    """
    获取Neo4j GraphRAG工具实例
    
    Args:
        uri: Neo4j URI
        username: 用户名
        password: 密码
        database: 数据库名
        force_new: 是否强制创建新实例
    
    Returns:
        Neo4jGraphRAG: 工具实例
    """
    global _neo4j_graphrag_instance
    
    if _neo4j_graphrag_instance is None or force_new:
        _neo4j_graphrag_instance = Neo4jGraphRAG(uri, username, password, database)
    
    return _neo4j_graphrag_instance 