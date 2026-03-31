"""
使用大模型进行知识图谱抽取模块
"""

import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

from sqlalchemy.orm import Session
from neo4j import Driver

from app.models.model import Model
from app.models.graph import Graph
from app.providers.manager import provider_manager
from app.utils.neo4j_utils import Neo4jService, get_neo4j_service
from app.utils.config import get_neo4j_config

# 配置日志
logger = logging.getLogger(__name__)

class LLMKnowledgeExtractor:
    """使用大模型进行知识抽取的工具类"""
    
    def __init__(
        self, 
        db: Session,
        neo4j_service: Optional[Neo4jService] = None
    ):
        """
        初始化LLM知识抽取器
        
        Args:
            db: 数据库会话
            neo4j_service: Neo4j服务实例，如果为None则自动创建
        """
        self.db = db
        
        # 初始化Neo4j服务
        if neo4j_service:
            self.neo4j_service = neo4j_service
        else:
            # 获取Neo4j配置
            neo4j_config = get_neo4j_config()
            if neo4j_config:
                self.neo4j_service = get_neo4j_service(
                    uri=neo4j_config.get("uri"),
                    username=neo4j_config.get("username"),
                    password=neo4j_config.get("password"),
                    database=neo4j_config.get("database", "neo4j")
                )
            else:
                self.neo4j_service = None
                logger.warning("未找到Neo4j配置，Neo4j服务未初始化")
    
    async def extract_knowledge(
        self,
        graph: Graph,
        text: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        使用大模型和Neo4j-GraphRAG从文本中抽取知识
        
        Args:
            graph: 知识图谱对象
            text: 文本内容
            schema: 图谱Schema定义
            
        Returns:
            Dict: 抽取结果
        """
        if not self.neo4j_service or not self.neo4j_service.driver:
            return {"success": False, "error": "Neo4j服务未初始化"}
        
        try:
            # 1. 检查图谱是否有关联的模型
            llm_model = None
            embedding_model = None
            
            # 获取图谱关联的对话模型
            if graph.model_id:
                llm_model = self.db.query(Model).filter(
                    Model.id == graph.model_id,
                    Model.type == "chat",
                    Model.status == "active"
                ).first()
            
            # 获取系统默认的embedding模型(任意选择一个活跃的embedding模型)
            embedding_model = self.db.query(Model).filter(
                Model.type == "embedding",
                Model.status == "active"
            ).first()
            
            # 确保有可用的模型
            if not llm_model:
                return {"success": False, "error": "未找到有效的对话模型"}
            
            # 2. 准备schema
            schema_entities = []
            schema_relations = []
            schema_triplets = []
            schema_definition = ""
            
            if schema:
                # 提取实体类型
                if "entityTypes" in schema and schema["entityTypes"]:
                    schema_entities = schema["entityTypes"]
                    schema_definition += "\n实体类型:\n"
                    for entity in schema_entities:
                        entity_name = entity.get("name", "未知实体")
                        entity_desc = entity.get("description", "")
                        schema_definition += f"- {entity_name}"
                        if entity_desc:
                            schema_definition += f": {entity_desc}"
                        schema_definition += "\n"
                        
                        # 添加属性信息
                        if "properties" in entity and entity["properties"]:
                            for prop in entity["properties"]:
                                prop_name = prop.get("name", "未知属性")
                                prop_type = prop.get("type", "STRING")
                                schema_definition += f"  * 属性: {prop_name} ({prop_type})\n"
                
                # 提取关系类型
                if "relationTypes" in schema and schema["relationTypes"]:
                    schema_relations = schema["relationTypes"]
                    schema_definition += "\n关系类型:\n"
                    for relation in schema_relations:
                        relation_name = relation.get("name", "未知关系")
                        relation_desc = relation.get("description", "")
                        
                        # 获取源和目标实体
                        source_type = relation.get("sourceType")
                        target_type = relation.get("targetType")
                        
                        source_entity = next((e for e in schema_entities if str(e.get("id")) == str(source_type)), None)
                        target_entity = next((e for e in schema_entities if str(e.get("id")) == str(target_type)), None)
                        
                        source_name = source_entity.get("name", "未知") if source_entity else "未知"
                        target_name = target_entity.get("name", "未知") if target_entity else "未知"
                        
                        schema_definition += f"- {relation_name} (从 {source_name} 到 {target_name})"
                        if relation_desc:
                            schema_definition += f": {relation_desc}"
                        schema_definition += "\n"
                        
                        # 添加关系三元组
                        if source_entity and target_entity:
                            schema_triplets.append((source_name, relation_name, target_name))
                            
                        # 添加属性信息
                        if "properties" in relation and relation["properties"]:
                            for prop in relation["properties"]:
                                prop_name = prop.get("name", "未知属性")
                                prop_type = prop.get("type", "STRING")
                                schema_definition += f"  * 属性: {prop_name} ({prop_type})\n"
            
            # 3. 调用大模型进行知识抽取
            # 构建提示词
            prompt = self._build_extraction_prompt(text, schema_definition, graph.dynamic_schema)
            
            # 准备调用大模型
            provider = provider_manager.get_provider(llm_model.provider)
            if not provider:
                return {"success": False, "error": f"找不到提供商: {llm_model.provider}"}
            
            # 构建消息
            messages = [
                {"role": "system", "content": "你是一个专业的知识图谱构建助手，擅长从文本中抽取实体、关系和属性。"},
                {"role": "user", "content": prompt}
            ]
            
            # 调用大模型
            response = await provider.chat_completion(
                api_key=llm_model.api_key,
                base_url=llm_model.base_url,
                model=llm_model.name,
                messages=messages,
                temperature=0.2,  # 低温度，使结果更确定性
                max_tokens=4000,
                stream=False
            )
            
            # 4. 解析大模型输出
            entities = []
            relations = []
            extraction_result = None  # 初始化变量
            print(f"大模型输出: {response}")
            if response and "choices" in response and response["choices"]:
                model_output = response["choices"][0]["message"]["content"]
                
                # 解析模型输出为知识图谱元素
                extraction_result = self._parse_model_output(model_output)
                
                if extraction_result:
                    entities = extraction_result.get("entities", [])
                    relations = extraction_result.get("relations", [])
            
            # 如果允许动态更新schema，根据抽取结果更新schema
            if graph.dynamic_schema and extraction_result:
                updated_schema = self._update_schema_from_extraction(
                    db=self.db,
                    graph=graph,
                    extraction_result=extraction_result,
                    original_schema=schema
                )
                # 更新schema变量，用于后续的Neo4j写入
                schema = updated_schema
                # 重新获取schema实体和关系
                schema_entities = schema.get("entityTypes", [])
                schema_relations = schema.get("relationTypes", [])
                schema_triplets = []
                
                # 重新构建schema_triplets
                for relation in schema_relations:
                    source_type = relation.get("sourceType")
                    target_type = relation.get("targetType")
                    
                    source_entity = next((e for e in schema_entities if str(e.get("id")) == str(source_type)), None)
                    target_entity = next((e for e in schema_entities if str(e.get("id")) == str(target_type)), None)
                    
                    if source_entity and target_entity:
                        source_name = source_entity.get("name", "未知")
                        target_name = target_entity.get("name", "未知")
                        relation_name = relation.get("name", "未知关系")
                        schema_triplets.append((source_name, relation_name, target_name))
            
            # 5. 将知识写入Neo4j
            graph_id = graph.id
            
            # 将知识写入Neo4j
            neo4j_result = self._write_to_neo4j(
                graph_id=graph_id,
                entities=entities,
                relations=relations,
                schema_entities=schema_entities,
                schema_relations=schema_relations,
                schema_triplets=schema_triplets
            )
            
            # 6. 返回结果
            return {
                "success": True,
                "message": "知识抽取完成",
                "entityCount": len(entities),
                "relationCount": len(relations),
                "neo4jResult": neo4j_result
            }
            
        except Exception as e:
            logger.error(f"大模型知识抽取失败: {str(e)}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"知识抽取失败: {str(e)}"
            }
    
    def _build_extraction_prompt(self, text: str, schema_definition: str = "", dynamic_schema: bool = True) -> str:
        """
        构建知识抽取提示词
        
        Args:
            text: 文本内容
            schema_definition: Schema定义
            dynamic_schema: 是否允许动态更新schema
            
        Returns:
            str: 提示词
        """
        prompt = "我希望你帮我从以下文本中抽取知识图谱的实体和关系，并以JSON格式输出。\n\n"
        
        # 添加schema约束
        if schema_definition:
            if dynamic_schema:
                prompt += f"""请参考以下已有的知识图谱Schema进行抽取但不要局限于此，并积极探索文本中可能存在的新实体类型和关系类型：
{schema_definition}

你的任务不仅是抽取符合已有schema的知识，更重要的是发现新的实体类型、关系类型和属性：
1. 积极识别文本中可能存在的、但schema中尚未定义的新实体类型
2. 挖掘实体之间可能存在的新关系类型
3. 为实体和关系提取更丰富的属性
4. 确保新发现的实体类型和关系类型命名规范、描述准确

请特别注意文本中描述的：
- 每个实体的特征属性和关系的特征属性

"""
            else:
                prompt += f"请严格按照以下知识图谱Schema进行抽取，不要创建Schema中未定义的实体类型和关系类型：\n{schema_definition}\n\n"
        else:
            prompt += """由于没有提供Schema，请自由抽取实体、关系和属性，但请确保：
1. 识别文本中的重要实体类型，如人物、组织、地点、事件、概念、产品等
2. 识别实体之间的各种关系类型，包括但不限于：从属关系、创建关系、参与关系、地理位置关系等
3. 抽取实体的详细属性信息
4. 实体类型和关系类型的命名应规范、描述性强，便于理解
5. 同一类型的实体或关系应保持一致的命名
"""
        
        prompt += "\n文本内容：\n\n```\n" + text + "\n```\n\n"
        
        prompt += """请将抽取结果以以下JSON格式返回：
```json
{
  "entities": [
    {
      "id": "唯一ID",
      "type": "实体类型",
      "name": "实体名称",
      "properties": {
        "属性名1": "属性值1",
        "属性名2": "属性值2"
      }
    }
  ],
  "relations": [
    {
      "source": "源实体ID（对应entities中的id）",
      "target": "目标实体ID（对应entities中的id）",
      "type": "关系类型",
      "properties": {
        "属性名1": "属性值1"
      }
    }
  ]
}
```

注意：
1. 每个实体必须有唯一的id、type和name
2. 关系的source和target必须对应entities中定义的实体id
3. 尽量提取完整的实体属性
4. 只返回JSON数据，不要有其他解释性文字
"""
        
        return prompt
    
    def _parse_model_output(self, model_output: str) -> Dict[str, Any]:
        """
        解析模型输出
        
        Args:
            model_output: 模型输出文本
            
        Returns:
            Dict: 解析后的实体和关系
        """
        try:
            # 尝试提取JSON部分
            json_content = model_output
            
            # 如果输出包含```json和```标记，提取它们之间的内容
            if "```json" in model_output:
                start_idx = model_output.find("```json") + 7
                end_idx = model_output.find("```", start_idx)
                if end_idx > start_idx:
                    json_content = model_output[start_idx:end_idx].strip()
            elif "```" in model_output:
                start_idx = model_output.find("```") + 3
                end_idx = model_output.find("```", start_idx)
                if end_idx > start_idx:
                    json_content = model_output[start_idx:end_idx].strip()
            
            # 解析JSON
            result = json.loads(json_content)
            
            # 确保结果包含必要的字段
            if "entities" not in result:
                result["entities"] = []
            if "relations" not in result:
                result["relations"] = []
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"解析模型输出失败 - JSON解析错误: {str(e)}")
            logger.error(f"模型原始输出: {model_output}")
            return {"entities": [], "relations": []}
        
        except Exception as e:
            logger.error(f"解析模型输出失败: {str(e)}")
            logger.error(f"模型原始输出: {model_output}")
            return {"entities": [], "relations": []}
    
    def _write_to_neo4j(
        self,
        graph_id: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        schema_entities: List[Dict[str, Any]] = None,
        schema_relations: List[Dict[str, Any]] = None,
        schema_triplets: List[Tuple[str, str, str]] = None
    ) -> Dict[str, Any]:
        """
        将知识写入Neo4j
        
        Args:
            graph_id: 图谱ID
            entities: 实体列表
            relations: 关系列表
            schema_entities: Schema实体类型
            schema_relations: Schema关系类型
            schema_triplets: Schema三元组
            
        Returns:
            Dict: 操作结果
        """
        if not self.neo4j_service or not self.neo4j_service.driver:
            return {"success": False, "error": "Neo4j服务未初始化"}
        
        try:
            with self.neo4j_service.driver.session(database=self.neo4j_service.database) as session:
                # 1. 创建实体
                entity_map = {}  # 映射模型ID到Neo4j节点ID
                
                for entity in entities:
                    entity_id = entity.get("id", str(uuid.uuid4()))
                    entity_type = entity.get("type", "Entity")
                    entity_name = entity.get("name", f"实体_{entity_id}")
                    properties = entity.get("properties", {})
                    
                    # 合并属性
                    neo4j_props = {
                        "id": entity_id,
                        "graph_id": graph_id,
                        "name": entity_name,
                        "created_at": "datetime()",
                        "source": "llm_extracted"
                    }
                    
                    # 添加其他属性
                    for key, value in properties.items():
                        if key not in neo4j_props:
                            neo4j_props[key] = value
                    
                    # 转换属性为Cypher参数
                    props_str = ", ".join([f"{k}: ${k}" for k in neo4j_props.keys()])
                    
                    # 创建实体节点
                    result = session.run(f"""
                    CREATE (e:{entity_type} {{{props_str}}})
                    RETURN id(e) as neo4j_id
                    """, **neo4j_props)
                    
                    record = result.single()
                    if record:
                        neo4j_id = record["neo4j_id"]
                        entity_map[entity_id] = neo4j_id
                
                # 2. 创建关系
                for relation in relations:
                    source_id = relation.get("source")
                    target_id = relation.get("target")
                    relation_type = relation.get("type", "RELATED_TO")
                    properties = relation.get("properties", {})
                    
                    # 检查源和目标实体是否存在
                    if source_id not in entity_map or target_id not in entity_map:
                        logger.warning(f"关系的源或目标实体不存在: {source_id} -> {target_id}")
                        continue
                    
                    # 合并属性
                    neo4j_props = {
                        "graph_id": graph_id,
                        "created_at": "datetime()",
                        "source": "llm_extracted"
                    }
                    
                    # 添加其他属性
                    for key, value in properties.items():
                        if key not in neo4j_props:
                            neo4j_props[key] = value
                    
                    # 转换属性为Cypher参数
                    props_str = ", ".join([f"{k}: ${k}" for k in neo4j_props.keys()])
                    
                    # 创建关系
                    session.run(f"""
                    MATCH (a) WHERE id(a) = $source_id
                    MATCH (b) WHERE id(b) = $target_id
                    CREATE (a)-[r:{relation_type} {{{props_str}}}]->(b)
                    """, 
                        source_id=entity_map[source_id], 
                        target_id=entity_map[target_id],
                        **neo4j_props
                    )
                
                # 4. 更新统计信息
                stats = self.neo4j_service.get_subgraph_statistics(graph_id)
                
                return {
                    "success": True,
                    "message": "知识已写入Neo4j",
                    "entityCount": len(entities),
                    "relationCount": len(relations),
                    "stats": stats
                }
        
        except Exception as e:
            logger.error(f"写入Neo4j失败: {str(e)}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"写入Neo4j失败: {str(e)}"
            }
    
    def _update_schema_from_extraction(
        self,
        db: Session,
        graph: Graph,
        extraction_result: Dict[str, Any],
        original_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        根据抽取结果更新图谱schema
        
        Args:
            db: 数据库会话
            graph: 图谱对象
            extraction_result: 抽取结果，包含entities和relations
            original_schema: 原始schema
            
        Returns:
            Dict: 更新后的schema
        """
        if not graph or not extraction_result:
            return original_schema or {}
        
        # 如果图谱不允许动态更新schema，则直接返回原始schema
        if not graph.dynamic_schema:
            return original_schema or {}
        
        # 获取原始schema，如果没有则创建一个新的
        schema = original_schema.copy() if original_schema else {}
        
        # 确保schema中包含必要的字段
        if "entityTypes" not in schema:
            schema["entityTypes"] = []
        if "relationTypes" not in schema:
            schema["relationTypes"] = []
        if "properties" not in schema:
            schema["properties"] = []
        
        # 现有实体和关系类型集合，用于去重
        existing_entity_types = {entity.get("name", "").lower() for entity in schema["entityTypes"]}
        existing_relation_types = {relation.get("name", "").lower() for relation in schema["relationTypes"]}
        
        # 从抽取结果中提取实体类型
        entities = extraction_result.get("entities", [])
        relations = extraction_result.get("relations", [])
        
        # 临时存储新的实体和关系类型映射
        new_entity_types = {}
        new_relation_types = {}
        
        # 处理实体类型
        for entity in entities:
            entity_type = entity.get("type")
            if not entity_type:
                continue
                
            entity_type_lower = entity_type.lower()
            
            # 如果是新的实体类型
            if entity_type_lower not in existing_entity_types and entity_type_lower not in new_entity_types:
                # 创建新的实体类型
                new_entity_type = {
                    "id": str(uuid.uuid4()),
                    "name": entity_type,
                    "description": f"自动从文本抽取的{entity_type}实体类型",
                    "color": self._get_random_color(),  # 随机颜色
                    "icon": "el-icon-info",  # 默认图标
                    "properties": [],
                    "requiredProperties": []
                }
                
                # 添加到schema
                schema["entityTypes"].append(new_entity_type)
                existing_entity_types.add(entity_type_lower)
                new_entity_types[entity_type_lower] = new_entity_type
                
                logger.info(f"为图谱 {graph.id} 添加新实体类型: {entity_type}")
            
            # 处理实体属性
            properties = entity.get("properties", {})
            if properties and (entity_type_lower in existing_entity_types or entity_type_lower in new_entity_types):
                # 找到对应的实体类型
                target_entity_type = next(
                    (e for e in schema["entityTypes"] if e.get("name", "").lower() == entity_type_lower), 
                    new_entity_types.get(entity_type_lower)
                )
                
                if target_entity_type:
                    # 确保实体类型有properties字段
                    if "properties" not in target_entity_type:
                        target_entity_type["properties"] = []
                    
                    # 现有属性名称集合
                    existing_properties = {p.get("name", "").lower() for p in target_entity_type["properties"]}
                    
                    # 添加新属性
                    for prop_name, prop_value in properties.items():
                        prop_name_lower = prop_name.lower()
                        if prop_name_lower not in existing_properties:
                            # 猜测属性类型
                            prop_type = self._guess_property_type(prop_value)
                            
                            # 创建新属性
                            new_property = {
                                "id": str(uuid.uuid4()),
                                "name": prop_name,
                                "type": prop_type,
                                "description": f"自动从文本抽取的{prop_name}属性"
                            }
                            
                            # 添加到实体类型
                            target_entity_type["properties"].append(new_property)
                            existing_properties.add(prop_name_lower)
                            
                            logger.info(f"为实体类型 {entity_type} 添加新属性: {prop_name} ({prop_type})")
        
        # 处理关系类型
        for relation in relations:
            relation_type = relation.get("type")
            if not relation_type:
                continue
                
            relation_type_lower = relation_type.lower()
            
            # 如果是新的关系类型
            if relation_type_lower not in existing_relation_types and relation_type_lower not in new_relation_types:
                # 从抽取结果中获取源和目标实体ID
                source_id = relation.get("source")
                target_id = relation.get("target")
                
                if not source_id or not target_id:
                    continue
                
                # 查找源和目标实体
                source_entity = next((e for e in entities if e.get("id") == source_id), None)
                target_entity = next((e for e in entities if e.get("id") == target_id), None)
                
                if not source_entity or not target_entity:
                    continue
                
                # 获取源和目标实体类型
                source_type = source_entity.get("type")
                target_type = target_entity.get("type")
                
                if not source_type or not target_type:
                    continue
                
                # 查找或创建源和目标实体类型ID
                source_type_id = None
                target_type_id = None
                
                # 在现有实体类型中查找
                for entity_type in schema["entityTypes"]:
                    if entity_type.get("name", "").lower() == source_type.lower():
                        source_type_id = entity_type.get("id")
                    if entity_type.get("name", "").lower() == target_type.lower():
                        target_type_id = entity_type.get("id")
                
                # 如果还没找到，检查新创建的实体类型
                if not source_type_id and source_type.lower() in new_entity_types:
                    source_type_id = new_entity_types[source_type.lower()].get("id")
                if not target_type_id and target_type.lower() in new_entity_types:
                    target_type_id = new_entity_types[target_type.lower()].get("id")
                
                # 如果找到了源和目标实体类型ID，创建新的关系类型
                if source_type_id and target_type_id:
                    # 创建新的关系类型
                    new_relation_type = {
                        "id": str(uuid.uuid4()),
                        "name": relation_type,
                        "sourceType": source_type_id,
                        "targetType": target_type_id,
                        "description": f"自动从文本抽取的{relation_type}关系类型",
                        "properties": [],
                        "requiredProperties": []
                    }
                    
                    # 添加到schema
                    schema["relationTypes"].append(new_relation_type)
                    existing_relation_types.add(relation_type_lower)
                    new_relation_types[relation_type_lower] = new_relation_type
                    
                    logger.info(f"为图谱 {graph.id} 添加新关系类型: {relation_type} (从 {source_type} 到 {target_type})")
            
            # 处理关系属性
            properties = relation.get("properties", {})
            if properties and (relation_type_lower in existing_relation_types or relation_type_lower in new_relation_types):
                # 找到对应的关系类型
                target_relation_type = next(
                    (r for r in schema["relationTypes"] if r.get("name", "").lower() == relation_type_lower), 
                    new_relation_types.get(relation_type_lower)
                )
                
                if target_relation_type:
                    # 确保关系类型有properties字段
                    if "properties" not in target_relation_type:
                        target_relation_type["properties"] = []
                    
                    # 现有属性名称集合
                    existing_properties = {p.get("name", "").lower() for p in target_relation_type["properties"]}
                    
                    # 添加新属性
                    for prop_name, prop_value in properties.items():
                        prop_name_lower = prop_name.lower()
                        if prop_name_lower not in existing_properties:
                            # 猜测属性类型
                            prop_type = self._guess_property_type(prop_value)
                            
                            # 创建新属性
                            new_property = {
                                "id": str(uuid.uuid4()),
                                "name": prop_name,
                                "type": prop_type,
                                "description": f"自动从文本抽取的{prop_name}属性"
                            }
                            
                            # 添加到关系类型
                            target_relation_type["properties"].append(new_property)
                            existing_properties.add(prop_name_lower)
                            
                            logger.info(f"为关系类型 {relation_type} 添加新属性: {prop_name} ({prop_type})")
        
        # 如果有更新，更新图谱配置
        if schema != original_schema:
            # 更新图谱配置
            if not graph.config:
                graph.config = {}
                
            graph.config["schema"] = schema
            db.add(graph)
            db.commit()
            
            logger.info(f"已更新图谱 {graph.id} 的schema")
        
        return schema
    
    def _guess_property_type(self, value: Any) -> str:
        """
        根据属性值猜测属性类型
        
        Args:
            value: 属性值
            
        Returns:
            str: 猜测的属性类型
        """
        if value is None:
            return "string"
            
        if isinstance(value, bool):
            return "boolean"
            
        if isinstance(value, int):
            return "integer"
            
        if isinstance(value, float):
            return "number"
            
        if isinstance(value, (list, tuple)):
            return "array"
            
        if isinstance(value, dict):
            return "object"
            
        # 尝试判断日期时间
        if isinstance(value, str):
            # 如果包含URL特征
            if value.startswith(("http://", "https://", "www.")):
                return "url"
                
            # 如果符合邮箱格式
            if "@" in value and "." in value.split("@")[-1]:
                return "email"
                
            # 如果全是数字和连字符，可能是ID
            if all(c.isdigit() or c == "-" for c in value):
                return "id"
                
            # 如果包含日期分隔符并且能解析为日期
            import re
            if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", value):
                return "date"
        
        # 默认为字符串
        return "string"
    
    def _get_random_color(self) -> str:
        """
        生成随机颜色代码
        
        Returns:
            str: 十六进制颜色代码
        """
        import random
        colors = [
            "#1890ff",  # 蓝色
            "#52c41a",  # 绿色
            "#fa8c16",  # 橙色
            "#eb2f96",  # 粉色
            "#722ed1",  # 紫色
            "#fadb14",  # 黄色
            "#13c2c2",  # 青色
            "#f5222d"   # 红色
        ]
        return random.choice(colors) 