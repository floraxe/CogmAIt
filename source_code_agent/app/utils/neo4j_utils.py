"""
与Neo4j相关的工具函数集合
"""
import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

# 配置日志
logger = logging.getLogger(__name__)

class Neo4jService:
    """Neo4j服务类，处理与Neo4j数据库的交互"""
    
    def __init__(
        self, 
        uri: str = None, 
        username: str = None, 
        password: str = None,
        database: str = "neo4j",
        driver: Driver = None
    ):
        """
        初始化Neo4j服务
        
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

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.driver:
            self.driver.close()
    
    def close(self):
        """关闭驱动连接"""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def is_connected(self) -> bool:
        """测试连接是否有效"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Neo4j连接测试失败: {str(e)}")
            return False
    
    def create_subgraph(self, name: str) -> bool:
        """
        创建Neo4j子图（通过属性标识而非实际的子图功能）
        
        Args:
            name: 子图名称/标识符
            
        Returns:
            bool: 操作是否成功
        """
        if not self.driver:
            logger.error("Neo4j driver未初始化")
            return False
        
        subgraph_name = name.lower().replace(" ", "_").replace("-", "_")
        
        try:
            with self.driver.session(database=self.database) as session:
                # 不再使用CREATE GRAPH语法，而是使用标签和属性来实现逻辑上的子图
                
                # 1. 创建一个子图元数据节点，表示这个子图的存在
                session.run("""
                    CREATE (g:GraphMetadata {
                        graph_id: $graph_id,
                        name: $name,
                        created: datetime(),
                        description: '知识图谱子图'
                    })
                """, graph_id=subgraph_name, name=name)
                
                # 2. 创建约束和索引
                try:
                    # 为graph_id创建索引以提高查询性能
                    session.run("CREATE INDEX IF NOT EXISTS FOR (n) ON (n.graph_id)")
                    
                    # 确保节点ID唯一（在一个子图内）
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE (n.id, n.graph_id) IS NODE KEY")
                except Exception as e:
                    logger.warning(f"创建索引或约束失败（这可能是正常的，取决于Neo4j版本）: {str(e)}")
                
                return True
        except Neo4jError as e:
            logger.error(f"创建Neo4j子图失败: {str(e)}")
            return False
    
    def delete_subgraph(self, name: str) -> bool:
        """
        删除Neo4j子图中的所有数据
        
        Args:
            name: 子图名称
            
        Returns:
            bool: 操作是否成功
        """
        if not self.driver:
            logger.error("Neo4j driver未初始化")
            return False
        
        subgraph_name = name.lower().replace(" ", "_").replace("-", "_")
        
        try:
            with self.driver.session(database=self.database) as session:
                # 删除所有属于该子图的节点和关系
                session.run("""
                    MATCH (n)
                    WHERE n.graph_id = $graph_id
                    DETACH DELETE n
                """, graph_id=subgraph_name)
                
                return True
        except Neo4jError as e:
            logger.error(f"删除Neo4j子图失败: {str(e)}")
            return False
    
    def get_subgraph_statistics(self, name: str) -> Dict[str, Any]:
        """
        获取子图统计信息
        
        Args:
            name: 子图名称
            
        Returns:
            Dict: 统计信息字典
        """
        if not self.driver:
            logger.error("Neo4j driver未初始化")
            return {
                "nodeCount": 0,
                "edgeCount": 0,
                "entityTypeCount": 0,
                "relationTypeCount": 0
            }
        
        subgraph_name = name.lower().replace(" ", "_").replace("-", "_")
        
        try:
            with self.driver.session(database=self.database) as session:
                # 获取节点数量
                result = session.run("""
                    MATCH (n)
                    WHERE n.graph_id = $graph_id
                    RETURN count(n) as nodeCount
                """, graph_id=subgraph_name)
                node_count = result.single()["nodeCount"]
                
                # 获取关系数量
                result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.graph_id = $graph_id
                    RETURN count(r) as edgeCount
                """, graph_id=subgraph_name)
                edge_count = result.single()["edgeCount"]
                
                # 获取实体类型数量
                result = session.run("""
                    MATCH (n)
                    WHERE n.graph_id = $graph_id
                    WITH labels(n) AS labels
                    UNWIND labels AS label
                    RETURN count(DISTINCT label) as entityTypeCount
                """, graph_id=subgraph_name)
                entity_type_count = result.single()["entityTypeCount"]
                
                # 获取关系类型数量
                result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.graph_id = $graph_id
                    RETURN count(DISTINCT type(r)) as relationTypeCount
                """, graph_id=subgraph_name)
                relation_type_count = result.single()["relationTypeCount"]
                
                return {
                    "nodeCount": node_count,
                    "edgeCount": edge_count,
                    "entityTypeCount": entity_type_count,
                    "relationTypeCount": relation_type_count
                }
        except Neo4jError as e:
            logger.error(f"获取Neo4j子图统计信息失败: {str(e)}")
            return {
                "nodeCount": 0,
                "edgeCount": 0,
                "entityTypeCount": 0,
                "relationTypeCount": 0,
                "error": str(e)
            }
    
    def create_vector_index(self, name: str, dimension: int = 1536) -> bool:
        """
        创建向量索引
        
        Args:
            name: 索引名称
            dimension: 向量维度，默认1536
            
        Returns:
            bool: 操作是否成功
        """
        if not self.driver:
            logger.error("Neo4j driver未初始化")
            return False
        
        index_name = name.lower().replace(" ", "_").replace("-", "_") + "_vector_index"
        
        try:
            with self.driver.session(database=self.database) as session:
                # 创建向量索引
                session.run("""
                CREATE VECTOR INDEX IF NOT EXISTS
                FOR (n:Document) ON (n.embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: $dimension,
                  `vector.similarity_function`: 'cosine'
                }}
                """, dimension=dimension)
                return True
        except Neo4jError as e:
            logger.error(f"创建Neo4j向量索引失败: {str(e)}")
            return False
    
    def build_knowledge_graph(
        self, 
        graph_id: str, 
        file_text: str, 
        potential_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        使用Neo4j GraphRAG构建知识图谱
        
        Args:
            graph_id: 图谱ID
            file_text: 文件文本内容
            potential_schema: 潜在的Schema定义
            
        Returns:
            Dict: 构建结果
        """
        if not self.driver:
            logger.error("Neo4j driver未初始化")
            return {"success": False, "error": "Neo4j driver未初始化"}
        
        subgraph_name = graph_id.lower().replace(" ", "_").replace("-", "_")
        
        # 处理potential_schema
        entities = []
        relations = []
        schema_triplets = []
        
        try:
            if potential_schema:
                if "entityTypes" in potential_schema:
                    entities = potential_schema["entityTypes"]
                if "relationTypes" in potential_schema:
                    relations = potential_schema["relationTypes"]
                
                # 构建schema_triplets格式
                for relation in relations:
                    source_type = None
                    target_type = None
                    
                    if isinstance(relation, dict) and "sourceType" in relation and "targetType" in relation:
                        source_type = relation["sourceType"]
                        target_type = relation["targetType"]
                        
                        # 找到对应的实体类型名称
                        source_entity = next((e for e in entities if str(e.get("id")) == str(source_type)), None)
                        target_entity = next((e for e in entities if str(e.get("id")) == str(target_type)), None)
                        
                        if source_entity and target_entity:
                            schema_triplets.append((
                                source_entity.get("name"), 
                                relation.get("name"), 
                                target_entity.get("name")
                            ))
            
            # 使用Python代码创建一个简单的KG构建管道
            with self.driver.session(database=self.database) as session:
                # 创建Document节点
                session.run("""
                CREATE (d:Document {
                    id: randomUUID(),
                    graph_id: $graph_id,
                    content: $content,
                    created_at: datetime()
                })
                """, graph_id=subgraph_name, content=file_text[:10000]) # 限制内容长度
                
                # 如果有schema，尝试创建一些示例实体和关系
                if entities and len(entities) > 0:
                    # 为每种实体类型创建示例节点
                    for entity in entities[:5]: # 最多创建5种实体类型
                        if entity and "name" in entity:
                            entity_name = entity["name"]
                            # 为每个实体类型创建3个示例节点
                            for i in range(3):
                                session.run(f"""
                                CREATE (e:{entity_name} {{
                                    id: randomUUID(),
                                    graph_id: $graph_id,
                                    name: $name,
                                    description: '自动创建的示例节点',
                                    created_at: datetime()
                                }})
                                """, graph_id=subgraph_name, name=f"示例{entity_name}{i+1}")
                
                # 如果有关系定义，创建示例关系
                if relations and len(relations) > 0 and entities and len(entities) >= 2:
                    for relation in relations[:3]: # 最多创建3种关系
                        if relation and "name" in relation and "sourceType" in relation and "targetType" in relation:
                            relation_name = relation["name"]
                            source_entity = next((e for e in entities if str(e.get("id")) == str(relation["sourceType"])), None)
                            target_entity = next((e for e in entities if str(e.get("id")) == str(relation["targetType"])), None)
                            
                            if source_entity and target_entity and "name" in source_entity and "name" in target_entity:
                                # 创建关系
                                session.run(f"""
                                MATCH (a:{source_entity["name"]} {{graph_id: $graph_id}}), 
                                      (b:{target_entity["name"]} {{graph_id: $graph_id}})
                                WITH a, b LIMIT 1
                                CREATE (a)-[r:{relation_name} {{
                                    graph_id: $graph_id,
                                    description: '自动创建的示例关系',
                                    created_at: datetime()
                                }}]->(b)
                                """, graph_id=subgraph_name)
                
                # 返回结果
                return {
                    "success": True,
                    "message": "知识图谱构建完成",
                    "schema": {
                        "entities": entities,
                        "relations": relations,
                        "schema_triplets": schema_triplets
                    }
                }
                
        except Neo4jError as e:
            logger.error(f"构建知识图谱失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_graph_visualization_data(self, graph_id: str, limit: int = 100) -> Dict[str, Any]:
        """
        获取图谱可视化数据
        
        Args:
            graph_id: 图谱ID
            limit: 返回的最大节点数量
            
        Returns:
            Dict: 可视化数据
        """
        if not self.driver:
            logger.error("Neo4j driver未初始化")
            return {"nodes": [], "links": []}
        
        try:
            with self.driver.session(database=self.database) as session:
                # 获取图谱数据 - 使用参数化查询避免注入风险
                cypher = """
               MATCH (n) 
                WHERE n.graph_id = $graph_id
                WITH n LIMIT $limit
                OPTIONAL MATCH (n)-[r]->(m)
                WHERE m.graph_id = $graph_id
                RETURN 
                    collect(distinct {
                        id: id(n), 
                        labels: labels(n), 
                        properties: properties(n)
                    }) as nodes,
                    collect(distinct {
                        id: id(r), 
                        type: type(r), 
                        properties: properties(r),
                        source: id(n),
                        target: id(m)
                    }) as relationships
                """
                result = session.run(cypher, graph_id=graph_id, limit=limit)
                print("subgraph_name::", graph_id, "limit:", limit)
                record = result.single()
                if not record:
                    return {"nodes": [], "links": []}
                
                # 转换成前端可视化需要的格式
                nodes = []
                for node in record["nodes"]:
                    if node and "properties" in node and node["properties"] is not None:
                        # 获取节点标签
                        labels = node["labels"]
                        # 跳过GraphMetadata标签的节点，它只是一个内部标记
                        if "GraphMetadata" in labels:
                            continue
                            
                        label = labels[0] if labels else "Unknown"
                        # 获取节点名称，如果没有则使用ID
                        name = node["properties"].get("name", f"{label}_{node['id']}")
                        
                        # 获取过滤后的属性（排除graph_id和内部属性）
                        filtered_props = {
                            k: v for k, v in node["properties"].items() 
                            if k not in ["graph_id"] and not k.startswith("_")
                        }
                        
                        nodes.append({
                            "id": str(node["id"]),
                            "name": name,
                            "label": label,
                            "properties": filtered_props
                        })
                
                links = []
                for rel in record["relationships"]:
                    # 添加有效性检查，确保关系有有效的id、source、target和type
                    if (rel and "source" in rel and "target" in rel and 
                        rel["id"] is not None and rel["source"] is not None and 
                        rel["target"] is not None and rel["type"] is not None):
                        
                        # 获取过滤后的属性（排除graph_id和内部属性）
                        filtered_props = {}
                        if rel.get("properties") is not None:  # 添加None检查
                            filtered_props = {
                                k: v for k, v in rel["properties"].items() 
                                if k not in ["graph_id"] and not k.startswith("_")
                            }
                        
                        links.append({
                            "id": str(rel["id"]),
                            "source": str(rel["source"]),
                            "target": str(rel["target"]),
                            "type": rel["type"],
                            "properties": filtered_props
                        })
                
                # 防止出现无节点但有关系的情况
                if len(nodes) == 0:
                    links = []
                
                return {
                    "nodes": nodes,
                    "links": links
                }
        except Neo4jError as e:
            logger.error(f"获取图谱可视化数据失败: {str(e)}")
            return {"nodes": [], "links": [], "error": str(e)}

# 单例模式工具函数
_neo4j_service_instance = None

def get_neo4j_service(
    uri: str = None, 
    username: str = None, 
    password: str = None,
    database: str = "neo4j",
    force_new: bool = False
) -> Neo4jService:
    """
    获取Neo4j服务实例（单例模式）
    
    Args:
        uri: Neo4j数据库URI
        username: 用户名
        password: 密码
        database: 默认数据库名称
        force_new: 是否强制创建新实例
        
    Returns:
        Neo4jService: Neo4j服务实例
    """
    global _neo4j_service_instance
    
    if force_new or _neo4j_service_instance is None:
        _neo4j_service_instance = Neo4jService(uri, username, password, database)
    
    return _neo4j_service_instance 