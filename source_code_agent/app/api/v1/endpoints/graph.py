from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import uuid
import logging
import json

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.db.session import get_db
from app.utils.deps import get_current_active_user
from app.models.graph import Graph as GraphModel
from app.models.model import Model as ModelModel
from app.models.graph_file import GraphFile
from app.utils.graph import upload_file_to_graph, delete_graph_file, process_graph_file, extract_knowledge_from_file, extract_knowledge_with_neo4j, extract_knowledge_with_llm_task
from app.models.extraction_task import ExtractionTask
from app.utils.neo4j_utils import get_neo4j_service
from app.utils.neo4j_graphrag import get_neo4j_graphrag
from app.utils.config import get_neo4j_config
from app.utils.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.graph import Graph
from app.schemas.graph import GraphCreate, GraphUpdate
from app.db.session import SessionLocal as db_session

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()

# 创建知识图谱的请求模型
class GraphCreate(BaseModel):
    name: str
    description: Optional[str] = None
    status: Optional[str] = "active"
    model_id: Optional[str] = None  # 关联的模型ID
    dynamic_schema: Optional[bool] = True  # 是否允许动态更新schema

# 知识图谱响应模型
class GraphResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str = "active"
    nodeCount: int = 0
    edgeCount: int = 0
    entityCount: int = 0
    relationCount: int = 0
    model_id: Optional[str] = None  # 关联的模型ID
    created: datetime = datetime.now()
    modified: datetime = datetime.now()
    dynamic_schema: bool = True  # 是否允许动态更新schema

# 新增聊天请求模型
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

# 新增聊天响应模型
class ChatResponse(BaseModel):
    message: str
    response: str

# 添加新的请求模型
class GraphFileBase(BaseModel):
    """知识图谱文件基础模式"""
    original_filename: str
    file_size: int
    file_type: str

# 知识图谱文件列表响应模型
class GraphFileListResponse(BaseModel):
    """知识图谱文件列表响应"""
    total: int
    items: List[Dict[str, Any]]

# 添加Neo4j配置的请求模型
class Neo4jConfigRequest(BaseModel):
    """Neo4j配置请求模型"""
    uri: str
    username: str
    password: str
    database: Optional[str] = "neo4j"

# 添加Neo4j连接测试的响应模型
class Neo4jTestResponse(BaseModel):
    """Neo4j连接测试响应模型"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

@router.get("/")
async def get_graphs(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    name: Optional[str] = None,
    current_user = Depends(get_current_active_user),
):
    """
    获取知识图谱列表
    """
    # 查询数据库中的知识图谱
    query = db.query(GraphModel)
    
    # 如果有名称过滤条件
    if name:
        query = query.filter(GraphModel.name.ilike(f"%{name}%"))
    
    # 计算总数
    total = query.count()
    
    # 分页查询
    query = query.offset((page - 1) * limit).limit(limit)
    
    # 获取结果
    graphs = [graph.to_dict() for graph in query.all()]
    
    return {
        "total": total,
        "items": graphs
    }

@router.post("/", response_model=GraphResponse)
async def create_graph(
    graph_data: GraphCreate = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    创建知识图谱
    """
    print(f"接收到创建知识图谱请求: {graph_data}")
    
    try:
        # 创建新的知识图谱对象并添加到数据库
        new_graph = GraphModel(
            name=graph_data.name,
            description=graph_data.description or "",
            status=graph_data.status,
            entity_count=0,
            relation_count=0,
            model_id=graph_data.model_id,  # 添加模型ID
            config={},
            neo4j_status="pending",  # 初始状态为pending，等待创建Neo4j子图
            dynamic_schema=graph_data.dynamic_schema,  # 设置动态更新schema
            user_id=current_user.id  # 添加用户ID
        )
        
        # 添加到数据库
        db.add(new_graph)
        db.commit()
        # 刷新以获取自动生成的属性
        db.refresh(new_graph)
        
        graph_id = new_graph.id
        print(f"成功创建知识图谱: {graph_id}")
        
        # 尝试创建Neo4j子图（异步任务，不影响图谱创建）
        try:
            # 获取Neo4j配置
            neo4j_config = get_neo4j_config()
            
            # 创建Neo4j服务
            neo4j_service = get_neo4j_service(
                uri=neo4j_config["uri"],
                username=neo4j_config["username"],
                password=neo4j_config["password"],
                database=neo4j_config["database"],
                force_new=True
            )
            
            # 生成子图名称
            subgraph_name = f"{graph_id}"
            
            # 创建子图
            success = neo4j_service.create_subgraph(subgraph_name)
            
            if success:
                # 更新图谱信息
                new_graph.neo4j_subgraph = subgraph_name
                new_graph.neo4j_status = "created"
                
                # 创建向量索引
                neo4j_service.create_vector_index(subgraph_name)
                
                # 获取统计信息
                stats = neo4j_service.get_subgraph_statistics(subgraph_name)
                new_graph.neo4j_stats = stats
                
                # 保存到数据库
                db.add(new_graph)
                db.commit()
                db.refresh(new_graph)
                
                print(f"成功创建Neo4j子图: {subgraph_name}")
            else:
                print(f"创建Neo4j子图失败")
        except Exception as e:
            print(f"创建Neo4j子图时出错: {str(e)}")
            # 异常不会影响图谱创建，只记录日志
        
        # 返回创建的知识图谱
        return new_graph.to_dict()
    except Exception as e:
        db.rollback()
        print(f"创建知识图谱时出错: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建知识图谱失败: {str(e)}"
        )

@router.get("/{graph_id}")
async def get_graph_detail(
    graph_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取知识图谱详情
    """
    # 从数据库查询指定ID的知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 节点和边的数量
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    
    # 转换为字典并添加统计信息
    result = graph.to_dict()
    result["nodeCount"] = node_count
    result["edgeCount"] = edge_count
    
    # 添加模型ID
    if graph.model_id:
        result["model_id"] = graph.model_id
    
    return result

@router.put("/{graph_id}")
async def update_graph(
    graph_id: str,
    graph_data: dict = Body(...),  # 改为dict类型，接受任意字段
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    更新知识图谱
    """
    # 从数据库查询指定ID的知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查请求是否只包含status字段（状态更新）
    if len(graph_data) == 1 and "status" in graph_data:
        # 仅更新状态
        graph.status = graph_data["status"]
    else:
        # 完整更新 - 验证必填字段
        if "name" not in graph_data:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="名称字段(name)是必需的"
            )
        
        # 更新知识图谱属性
        graph.name = graph_data.get("name", graph.name)
        graph.description = graph_data.get("description", graph.description)
        graph.status = graph_data.get("status", graph.status)
        graph.model_id = graph_data.get("model_id", graph.model_id)
        
        # 如果提供了dynamic_schema字段，则更新
        if "dynamic_schema" in graph_data:
            graph.dynamic_schema = graph_data["dynamic_schema"]
    
    try:
        # 提交更改
        db.commit()
        db.refresh(graph)
        
        return graph.to_dict()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新知识图谱失败: {str(e)}"
        )

@router.delete("/{graph_id}")
async def delete_graph(
    graph_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    删除知识图谱
    """
    # 从数据库查询指定ID的知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    try:
        # 删除关联的文件记录，而不是尝试断开关联
        graph_files = db.query(GraphFile).filter(GraphFile.graph_id == graph_id).all()
        for file in graph_files:
            # 从文件系统删除文件
            if file.path and os.path.exists(file.path):
                try:
                    os.remove(file.path)
                except Exception as e:
                    # 文件删除失败不阻止数据库记录的删除
                    print(f"删除文件失败: {file.path}, 错误: {str(e)}")
            
            # 从数据库中删除文件记录
            db.delete(file)
        
        # 提交文件删除
        db.commit()
        
        # 然后删除知识图谱
        db.delete(graph)
        db.commit()
        
        return {"success": True, "message": "知识图谱已删除"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除知识图谱失败: {str(e)}"
        )

@router.post("/{graph_id}/test")
async def test_graph_connection(
    graph_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    测试知识图谱连接
    """
    # 从数据库查询指定ID的知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    
    if not graph:
        return {
            "status": "error",
            "message": "知识图谱不存在"
        }
    
    # 计算节点和边的数量
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    
    # 模拟测试响应时间
    import time
    start_time = time.time()
    time.sleep(0.1)  # 模拟处理时间
    response_time = int((time.time() - start_time) * 1000)
    
    return {
        "status": "success",
        "message": "连接成功",
        "response": {
            "nodeCount": node_count,
            "edgeCount": edge_count,
            "responseTime": response_time
        }
    }

@router.get("/{graph_id}/export")
async def export_graph(
    graph_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    导出知识图谱
    """
    # 临时实现
    return {"data": {}, "format": "json"}

@router.get("/{graph_id}/visualization")
async def get_graph_visualization(
    graph_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取知识图谱可视化数据
    """
    # 临时实现
    return {
        "nodes": [],
        "edges": []
    }

@router.get("/{graph_id}/nodes")
async def get_graph_nodes(
    graph_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取图谱节点
    """
    # 临时实现
    return []

@router.post("/{graph_id}/nodes")
async def add_graph_node(
    graph_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    添加图谱节点
    """
    # 临时实现
    return {"id": 1, "name": "新节点"}

@router.get("/{graph_id}/edges")
async def get_graph_edges(
    graph_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取图谱边
    """
    # 临时实现
    return []

@router.post("/{graph_id}/edges")
async def add_graph_edge(
    graph_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    添加图谱边
    """
    # 临时实现
    return {"id": 1, "source": 1, "target": 2, "label": "关联"} 

@router.get("/{graph_id}/schema")
async def get_graph_schema(
    graph_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取知识图谱Schema数据
    """
    # 从数据库查询指定ID的知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 首先检查是否已经有存储的schema数据
    if graph.config and "schema" in graph.config:
        print(f"从graph.config中读取schema: {graph.config['schema']}")
        return graph.config["schema"]
    
    # 如果没有存储的schema，则从节点和边进行分析构建
    print("无存储的schema数据，从节点和边分析构建")
    
    # 分析并构建Schema数据
    # 收集所有独特的节点类型
    node_types = {}
    # 收集所有独特的关系类型
    relation_types = {}
    # 收集所有属性
    properties = []
    
    # 处理节点类型
    for node in graph.nodes:
        if node.node_type not in node_types:
            node_types[node.node_type] = {
                "id": node.node_type,
                "name": node.node_type,
                "color": "#1890ff",  # 默认颜色
                "icon": "el-icon-user",  # 默认图标
                "description": "",
                "requiredProperties": []
            }
        
        # 收集该类型节点的所有属性
        if node.properties:
            for prop_name, prop_value in node.properties.items():
                # 这里可以添加属性收集逻辑
                pass
    
    # 处理关系类型
    for edge in graph.edges:
        rel_key = edge.relation
        if rel_key not in relation_types:
            # 获取源节点和目标节点的类型
            source_type = next((n.node_type for n in graph.nodes if n.id == edge.source_id), "未知")
            target_type = next((n.node_type for n in graph.nodes if n.id == edge.target_id), "未知")
            
            relation_types[rel_key] = {
                "id": rel_key,
                "name": edge.relation,
                "sourceType": source_type,
                "targetType": target_type,
                "color": "#52c41a",  # 默认颜色
                "description": "",
                "requiredProperties": []
            }
    
    # 构建最终的Schema数据
    schema_data = {
        "entityTypes": list(node_types.values()),
        "relationTypes": list(relation_types.values()),
        "properties": properties
    }
    
    return schema_data

@router.put("/{graph_id}/schema")
async def update_graph_schema(
    graph_id: str,
    schema_data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    更新知识图谱Schema
    """
    print(f"用户：： {current_user}")
    
    # 从数据库查询指定ID的知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查schema数据结构
    if "entityTypes" not in schema_data or "relationTypes" not in schema_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Schema数据格式不正确，必须包含entityTypes和relationTypes"
        )
    
    try:
        print(f"更新schema数据: {schema_data}")
        print(f"更新前图谱config: {graph.config}")
        
        # 确保config是一个有效的字典
        if graph.config is None:
            graph.config = {}
        
        # 修复大整数ID问题 - 将大整数ID转换为字符串
        # 处理实体类型ID
        for entity_type in schema_data["entityTypes"]:
            if "id" in entity_type and isinstance(entity_type["id"], int) and entity_type["id"] > 2147483647:
                entity_type["id"] = str(entity_type["id"])

        # 处理关系类型ID
        for relation_type in schema_data["relationTypes"]:
            if "id" in relation_type and isinstance(relation_type["id"], int) and relation_type["id"] > 2147483647:
                relation_type["id"] = str(relation_type["id"])
            if "sourceType" in relation_type and isinstance(relation_type["sourceType"], int) and relation_type["sourceType"] > 2147483647:
                relation_type["sourceType"] = str(relation_type["sourceType"])
            if "targetType" in relation_type and isinstance(relation_type["targetType"], int) and relation_type["targetType"] > 2147483647:
                relation_type["targetType"] = str(relation_type["targetType"])

        # 处理属性ID
        if "properties" in schema_data:
            for prop in schema_data["properties"]:
                if "id" in prop and isinstance(prop["id"], int) and prop["id"] > 2147483647:
                    prop["id"] = str(prop["id"])
        
        # 直接使用纯字典更新config
        new_config = dict(graph.config) if graph.config else {}
        new_config["schema"] = schema_data
        
        # 输出转换后的数据类型
        print(f"处理后的schema数据类型: {type(schema_data)}")
        print(f"new_config类型: {type(new_config)}")
        print(f"处理后的schema数据: {schema_data}")
        print(f"新的config: {new_config}")
        
        # 使用UPDATE语句直接更新数据库
        from sqlalchemy import text
        
        # 准备JSON字符串
        import json
        config_json = json.dumps(new_config)
        
        print(f"序列化后的JSON字符串: {config_json}")
        
        # 执行原生SQL更新
        update_query = text(f"""
            UPDATE graphs 
            SET config = :config
            WHERE id = :id
        """)
        
        result = db.execute(
            update_query, 
            {"config": config_json, "id": graph_id}
        )
        
        print(f"SQL更新结果: {result.rowcount} 行受影响")
        
        # 提交更改
        db.commit()
        print("数据库提交完成")
        
        # 重新查询数据库
        updated_graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
        print(f"重新查询后图谱config: {updated_graph.config}")
        
        if updated_graph.config and "schema" in updated_graph.config:
            print(f"Schema已成功保存: {updated_graph.config['schema']}")
            return {
                "success": True, 
                "message": "Schema更新成功",
                "config": updated_graph.config
            }
        else:
            print("警告: Schema似乎未成功保存")
            # 尝试使用SQLAlchemy ORM方式更新
            print("尝试使用ORM方式重新更新...")
            
            # 直接设置字典值
            updated_graph.config = new_config
            db.add(updated_graph)
            db.commit()
            db.refresh(updated_graph)
            
            print(f"ORM更新后图谱config: {updated_graph.config}")
            
            if updated_graph.config and "schema" in updated_graph.config:
                return {
                    "success": True, 
                    "message": "Schema通过ORM方式更新成功",
                    "config": updated_graph.config
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Schema未能成功保存到数据库"
                )
    except Exception as e:
        db.rollback()
        print(f"更新Schema时出错: {str(e)}")
        
        # 捕获详细错误信息
        import traceback
        error_traceback = traceback.format_exc()
        print(f"错误详情: {error_traceback}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新Schema失败: {str(e)}"
        )

# 添加新的接口 - 获取节点属性
@router.get("/{graph_id}/nodes/{node_id}/properties")
async def get_node_properties(
    graph_id: str,
    node_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取节点的属性
    """
    # 查询知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 查询节点
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="节点不存在"
        )
    
    # 返回节点属性
    return node.properties or {}

# 添加新的接口 - 更新节点属性
@router.put("/{graph_id}/nodes/{node_id}/properties")
async def update_node_properties(
    graph_id: str,
    node_id: str,
    properties: dict = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    更新节点的属性
    """
    # 查询知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 查询节点
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="节点不存在"
        )
    
    try:
        # 更新节点属性
        if node.properties is None:
            node.properties = {}
        
        # 合并新属性
        for key, value in properties.items():
            node.properties[key] = value
        
        # 保存到数据库
        db.commit()
        db.refresh(node)
        
        return {"success": True, "message": "属性更新成功", "properties": node.properties}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新属性失败: {str(e)}"
        )

# 添加新的接口 - 添加实体
@router.post("/{graph_id}/entities")
async def add_entity(
    graph_id: str,
    entity_data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    添加实体到知识图谱
    """
    # 查询知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    try:
        # 检查必要参数
        if "name" not in entity_data or "type" not in entity_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="实体数据必须包含name和type字段"
            )
        
        # 确保properties是字典
        properties = entity_data.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        
        # 创建新节点
        from app.models.graph import GraphNode
        
        new_node = GraphNode(
            graph_id=graph_id,
            name=entity_data["name"],
            node_type=entity_data["type"],
            properties=properties,
            description=entity_data.get("description", "")
        )
        
        # 添加到数据库
        db.add(new_node)
        db.commit()
        db.refresh(new_node)
        
        # 更新知识图谱的实体计数
        graph.entity_count = graph.entity_count + 1
        db.commit()
        
        return new_node.to_dict()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加实体失败: {str(e)}"
        )

# 添加新的接口 - 获取实体列表
@router.get("/{graph_id}/entities")
async def get_entities(
    graph_id: str,
    entity_type: Optional[str] = None,
    name: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取知识图谱中的实体列表
    """
    from sqlalchemy import or_
    from app.models.graph import GraphNode
    
    # 查询知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 构建查询
    query = db.query(GraphNode).filter(GraphNode.graph_id == graph_id)
    
    # 应用筛选条件
    if entity_type:
        query = query.filter(GraphNode.node_type == entity_type)
    
    if name:
        query = query.filter(GraphNode.name.ilike(f"%{name}%"))
    
    # 计算总数
    total = query.count()
    
    # 分页
    entities = query.offset(skip).limit(limit).all()
    
    # 转换为字典
    entity_list = [entity.to_dict() for entity in entities]
    
    return {
        "total": total,
        "items": entity_list
    }

# 添加新的接口 - 获取实体属性定义
@router.get("/{graph_id}/entity-types/{entity_type}/properties")
async def get_entity_type_properties(
    graph_id: str,
    entity_type: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取实体类型的属性定义
    """
    # 查询知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 获取schema
    schema = {}
    if graph.config and "schema" in graph.config:
        schema = graph.config["schema"]
    
    # 查找实体类型
    entity_types = schema.get("entityTypes", [])
    entity_type_def = next((et for et in entity_types if et.get("id") == entity_type or et.get("name") == entity_type), None)
    
    if not entity_type_def:
        return {
            "entityType": entity_type,
            "properties": []
        }
    
    # 获取该实体类型的属性定义
    properties = []
    property_ids = entity_type_def.get("requiredProperties", [])
    
    if schema.get("properties"):
        for prop in schema["properties"]:
            if prop.get("id") in property_ids:
                properties.append(prop)
    
    return {
        "entityType": entity_type_def,
        "properties": properties
    }

@router.post("/{graph_id}/chat", response_model=ChatResponse)
async def chat_with_graph(
    graph_id: str,
    chat_data: ChatRequest = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    与知识图谱进行对话
    """
    # 从数据库查询指定ID的知识图谱
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查知识图谱是否关联了模型
    if not graph.model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该知识图谱未关联对话模型，请先在图谱设置中关联一个对话模型"
        )
    
    try:
        # 获取关联的模型
        model = db.query(ModelModel).filter(ModelModel.id == graph.model_id).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="关联的对话模型不存在或已被删除"
            )
        
        # 检查模型状态
        if model.status != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="关联的对话模型当前处于非活跃状态，无法使用"
            )
        
        # 检查模型类型
        if model.type != "chat":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="关联的模型不是对话类型，无法用于聊天"
            )
        
        # TODO: 实现真实的知识图谱对话逻辑，这里仅返回模拟响应
        # 在实际实现中，应该:
        # 1. 调用模型进行对话，或通过模型提供商的API接口发送对话请求
        # 2. 根据图谱的数据和用户的问题生成答案
        # 3. 记录对话历史用于连续对话
        
        # 模拟响应
        response = f"您好，我已收到您的问题：\"{chat_data.message}\"。这是基于知识图谱\"{graph.name}\"的回答，使用了模型\"{model.name}\"。该知识图谱包含{graph.entity_count}个实体和{graph.relation_count}个关系。"
        
        return {
            "message": chat_data.message,
            "response": response
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"与知识图谱对话失败: {str(e)}"
        )

@router.post("/{graph_id}/files/upload")
async def upload_file(
    graph_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    上传单个或多个文件到知识图谱，支持异步处理
    
    参数:
        graph_id: 知识图谱ID
        files: 上传的文件列表
    """
    try:
        # 检查知识图谱是否存在
        graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识图谱不存在"
            )
        
        # 确保files始终是列表
        if not isinstance(files, list):
            files = [files]
        
        # 上传目录
        upload_dir = os.path.join(os.getcwd(), "uploads")
        
        # 处理每个文件
        uploaded_files = []
        for file in files:
            try:
                # 上传文件，传递当前用户ID
                db_file = await upload_file_to_graph(
                    db=db, 
                    graph_id=graph_id, 
                    file=file,
                    background_tasks=background_tasks,
                    upload_dir=upload_dir,
                    user_id=current_user.id  # 添加用户ID
                )
                
                uploaded_files.append(db_file.to_dict())
            except Exception as file_error:
                # 单个文件上传失败不应该导致整个请求失败
                print(f"文件 {file.filename} 上传失败: {str(file_error)}")
                uploaded_files.append({
                    "filename": file.filename,
                    "error": str(file_error),
                    "status": "failed"
                })
        
        return {
            "code": 200,
            "message": f"已上传 {len(uploaded_files)} 个文件，后台处理中",
            "data": {
                "files": uploaded_files
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"文件上传失败: {str(e)}"}
        )

@router.get("/{graph_id}/files", response_model=GraphFileListResponse)
async def read_graph_files(
    graph_id: str,
    db: Session = Depends(get_db),
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user = Depends(get_current_active_user),
):
    """
    获取知识图谱的文件列表
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 构建查询
    query = db.query(GraphFile).filter(GraphFile.graph_id == graph_id)
    
    # 如果有状态过滤
    if status:
        query = query.filter(GraphFile.status == status)
    
    # 计算总数
    total = query.count()
    
    # 分页查询
    files = query.order_by(GraphFile.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    
    # 转换为字典列表
    file_list = [file.to_dict() for file in files]
    
    return {
        "total": total,
        "items": file_list
    }

@router.delete("/{graph_id}/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    graph_id: str,
    file_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    删除知识图谱的文件
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 查询文件
    file = db.query(GraphFile).filter(
        GraphFile.id == file_id,
        GraphFile.graph_id == graph_id
    ).first()
    
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    try:
        # 删除文件
        delete_graph_file(db, file_id)
        return {"message": "文件已删除"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除文件失败: {str(e)}"
        )

@router.post("/{graph_id}/files/{file_id}/parse")
async def parse_file(
    graph_id: str,
    file_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    解析文件内容
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 查询文件
    file = db.query(GraphFile).filter(
        GraphFile.id == file_id,
        GraphFile.graph_id == graph_id
    ).first()
    
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 检查文件状态
    if file.status == "processing":
        return {
            "code": status.HTTP_400_BAD_REQUEST,
            "message": "文件正在处理中，请稍后再试"
        }
    
    try:
        # 更新文件状态为处理中
        file.status = "processing"
        db.add(file)
        db.commit()
        
        # 添加后台任务解析文件
        background_tasks.add_task(
            process_graph_file,
            file_id,
            graph_id
        )
        
        return {
            "code": status.HTTP_200_OK,
            "message": "文件解析任务已提交，请稍后查看结果",
            "data": {
                "fileId": file_id,
                "status": "processing"
            }
        }
    except Exception as e:
        # 恢复文件状态
        file.status = "failed"
        file.error = str(e)
        db.add(file)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"解析文件失败: {str(e)}"
        )

@router.get("/{graph_id}/files/{file_id}/status")
async def get_file_parse_status(
    graph_id: str,
    file_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取文件解析状态
    """
    # 查询文件
    file = db.query(GraphFile).filter(
        GraphFile.id == file_id,
        GraphFile.graph_id == graph_id
    ).first()
    
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    return {
        "fileId": file_id,
        "status": file.status,
        "error": file.error,
        "message": get_status_message(file.status)
    }

@router.post("/{graph_id}/extract")
async def extract_knowledge(
    graph_id: str,
    background_tasks: BackgroundTasks,
    data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    从文件内容提取知识图谱，直接使用关联的大模型
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查图谱是否关联了模型
    if not graph.model_id:
        return {
            "code": status.HTTP_400_BAD_REQUEST,
            "message": "知识图谱未关联模型，请先在基本信息中关联一个对话模型"
        }
    
    # 检查参数
    if "fileId" not in data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少必要参数: fileId"
        )
    
    file_id = data["fileId"]
    
    # 查询文件
    file = db.query(GraphFile).filter(
        GraphFile.id == file_id,
        GraphFile.graph_id == graph_id
    ).first()
    
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 检查文件状态：如果是pending状态，先触发解析流程
    if file.status == "pending" or file.status == "failed":
        # 更新文件状态为处理中
        file.status = "processing"
        db.add(file)
        db.commit()
        
        # 先执行文件解析任务
        background_tasks.add_task(
            process_graph_file,
            file_id,
            graph_id
        )

    # 创建一个知识抽取任务
    task = ExtractionTask(
        graph_id=graph_id,
        file_id=file_id,
        task_type="llm_extraction",
        status="pending",
        parameters={"use_llm": True},
        result=None,
        model_id=graph.model_id
    )
    
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # 添加后台任务：等待文件解析完成后执行知识抽取
    background_tasks.add_task(
        extract_knowledge_with_llm_task,  # 直接调用LLM知识抽取任务
        db_session_maker=db_session, 
        task_id=task.id,
        graph_id=graph_id,
        file_id=file_id
    )
    
    return {
        "code": status.HTTP_200_OK,
        "message": "大模型知识抽取任务已提交",
        "data": {
            "taskId": task.id,
            "fileId": file_id,
            "status": "pending"
        }
    }

@router.post("/{graph_id}/extraction-tasks/{task_id}/retry")
async def retry_extraction_task(
    graph_id: str,
    task_id: str,
    data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_active_user),
):
    """
    重新抽取知识（用于抽取失败后重试）
    """
    try:
        # 检查图谱是否存在
        graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识图谱不存在"
            )
        
        # 检查任务是否存在
        task = db.query(ExtractionTask).filter(
            ExtractionTask.id == task_id,
            ExtractionTask.graph_id == graph_id
        ).first()
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="抽取任务不存在"
            )
        
        # 只有失败的任务才能重试
        if task.status != "failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"只有失败的任务才能重试，当前任务状态: {task.status}"
            )
        
        # 更新任务状态
        task.status = "pending"
        task.message = "任务重新提交"
        task.retry_count = (task.retry_count or 0) + 1
        task.updated_at = datetime.now()
        db.commit()
        
        # 获取文件信息
        file = db.query(GraphFile).filter(GraphFile.id == task.file_id).first()
        if not file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文件不存在"
            )
        
        # 准备抽取参数
        extraction_params = {
            "fileId": task.file_id,
            "modelId": task.model_id,
            "prompt": data.get("prompt") or task.prompt,
            "parameters": data.get("parameters") or task.parameters
        }
        
        # 在后台启动抽取任务
        background_tasks.add_task(
            extract_knowledge_worker,
            db,
            graph_id,
            task_id,
            extraction_params
        )
        
        return {
            "message": "知识抽取任务已重新提交",
            "taskId": task_id
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"重新提交知识抽取任务失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重新提交知识抽取任务失败: {str(e)}"
        )

# 辅助函数：根据状态获取消息
def get_status_message(status):
    status_messages = {
        "uploading": "文件上传中",
        "uploaded": "文件已上传",
        "processing": "文件处理中",
        "parsing": "文件解析中",
        "parsed": "文件已解析",
        "extracting": "知识抽取中",
        "extracted": "知识已抽取",
        "completed": "处理完成",
        "failed": "处理失败"
    }
    return status_messages.get(status, "未知状态")

@router.get("/{graph_id}/extraction-tasks")
async def get_extraction_tasks(
    graph_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取知识图谱的抽取任务列表
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 查询与该图谱关联的所有抽取任务
    tasks = db.query(ExtractionTask).filter(
        ExtractionTask.graph_id == graph_id
    ).order_by(ExtractionTask.created_at.desc()).all()
    
    # 转换为字典列表
    result = []
    for task in tasks:
        # 获取关联的文件信息
        file_info = {}
        if task.file_id:
            file = db.query(GraphFile).filter(GraphFile.id == task.file_id).first()
            if file:
                file_info = {
                    "id": file.id,
                    "filename": file.filename,
                    "originalFilename": file.original_filename,
                    "status": file.status
                }
        
        # 添加任务信息
        task_dict = task.to_dict()
        task_dict["file"] = file_info
        
        # 如果有实体和关系计数，添加到结果中
        if task.result:
            task_dict["entityCount"] = task.result.get("entities", 0)
            task_dict["relationCount"] = task.result.get("relations", 0)
        
        result.append(task_dict)
    
    return result

@router.get("/{graph_id}/extraction-tasks/{task_id}")
async def get_extraction_task_detail(
    graph_id: str,
    task_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取知识抽取任务详情
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 查询任务
    task = db.query(ExtractionTask).filter(
        ExtractionTask.id == task_id,
        ExtractionTask.graph_id == graph_id
    ).first()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="抽取任务不存在"
        )
    
    # 获取关联的文件信息
    file_info = {}
    if task.file_id:
        file = db.query(GraphFile).filter(GraphFile.id == task.file_id).first()
        if file:
            file_info = {
                "id": file.id,
                "filename": file.filename,
                "originalFilename": file.original_filename,
                "status": file.status
            }
    
    # 构建详细信息
    detail = task.to_dict()
    detail["file"] = file_info
    
    return detail

@router.get("/{graph_id}/extraction-tasks/{task_id}/result")
async def get_extraction_task_result(
    graph_id: str,
    task_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取知识抽取任务结果
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 查询任务
    task = db.query(ExtractionTask).filter(
        ExtractionTask.id == task_id,
        ExtractionTask.graph_id == graph_id
    ).first()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="抽取任务不存在"
        )
    
    # 如果任务未完成，返回状态
    if task.status != "completed":
        return {
            "status": task.status,
            "message": task.message or get_status_message(task.status),
            "entityCount": 0,
            "relationCount": 0,
            "data": {}
        }
    
    # 构建结果
    result = {
        "status": "completed",
        "entityCount": task.entity_count or task.result.get("entities", 0) if task.result else 0,
        "relationCount": task.relation_count or task.result.get("relations", 0) if task.result else 0,
        "data": task.result or {}
    }
    
    return result

@router.post("/{graph_id}/extraction-tasks/{task_id}/cancel")
async def cancel_extraction_task(
    graph_id: str,
    task_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    取消知识抽取任务
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 查询任务
    task = db.query(ExtractionTask).filter(
        ExtractionTask.id == task_id,
        ExtractionTask.graph_id == graph_id
    ).first()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="抽取任务不存在"
        )
    
    # 只有进行中或等待中的任务才能取消
    if task.status not in ["pending", "processing", "running"]:
        return {
            "code": status.HTTP_400_BAD_REQUEST,
            "message": f"只有进行中或等待中的任务才能取消，当前任务状态: {task.status}"
        }
    
    # 更新任务状态
    task.status = "cancelled"
    task.message = "任务已取消"
    task.updated_at = datetime.now()
    db.add(task)
    db.commit()
    
    return {
        "code": status.HTTP_200_OK,
        "message": "任务已取消",
        "data": {
            "taskId": task.id,
            "status": "cancelled"
        }
    }

@router.get("/{graph_id}/files/{file_id}/preview")
async def get_file_preview(
    graph_id: str,
    file_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取文件文本内容预览
    """
    # 查询文件
    file = db.query(GraphFile).filter(
        GraphFile.id == file_id,
        GraphFile.graph_id == graph_id
    ).first()
    
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 如果文件未解析或解析失败
    if file.status not in ["parsed", "completed", "extracted"]:
        return {
            "status": file.status,
            "message": "文件未解析或解析失败",
            "content": None
        }
    
    # 如果没有文本内容
    if not file.text_content:
        return {
            "status": file.status,
            "message": "文件没有可用的文本内容",
            "content": None
        }
    
    # 返回文件文本内容
    return {
        "status": file.status,
        "message": "获取文件内容成功",
        "content": file.text_content
    }

@router.post("/{graph_id}/neo4j-test")
async def test_neo4j_connection(
    graph_id: str,
    config: Optional[Neo4jConfigRequest] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    测试Neo4j连接
    """
    # 获取图谱信息
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 获取Neo4j配置
    if config:
        # 使用请求中的配置
        neo4j_config = {
            "uri": config.uri,
            "username": config.username,
            "password": config.password,
            "database": config.database
        }
    else:
        # 使用系统配置
        neo4j_config = get_neo4j_config()
    
    try:
        # 创建Neo4j服务并测试连接
        neo4j_service = get_neo4j_service(
            uri=neo4j_config["uri"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            database=neo4j_config["database"],
            force_new=True
        )
        
        # 测试连接
        connected = neo4j_service.is_connected()
        
        if connected:
            # 检查子图是否存在
            has_subgraph = False
            if graph.neo4j_subgraph:
                # TODO: 检查子图是否存在的逻辑
                has_subgraph = True
            
            # 构建响应
            return {
                "status": "success",
                "message": "Neo4j连接成功",
                "data": {
                    "connected": True,
                    "has_subgraph": has_subgraph,
                    "subgraph_name": graph.neo4j_subgraph if graph.neo4j_subgraph else None,
                    "neo4j_status": graph.neo4j_status
                }
            }
        else:
            return {
                "status": "error",
                "message": "Neo4j连接失败",
                "data": {
                    "connected": False
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Neo4j连接测试失败: {str(e)}",
            "data": {
                "connected": False,
                "error": str(e)
            }
        }

@router.post("/{graph_id}/neo4j-subgraph")
async def create_neo4j_subgraph(
    graph_id: str,
    config: Optional[Neo4jConfigRequest] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    创建Neo4j子图
    """
    # 获取图谱信息
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 获取Neo4j配置
    if config:
        # 使用请求中的配置
        neo4j_config = {
            "uri": config.uri,
            "username": config.username,
            "password": config.password,
            "database": config.database
        }
    else:
        # 使用系统配置
        neo4j_config = get_neo4j_config()
    
    try:
        # 创建Neo4j服务
        neo4j_service = get_neo4j_service(
            uri=neo4j_config["uri"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            database=neo4j_config["database"],
            force_new=True
        )
        
        # 生成子图名称
        subgraph_name = f"graph_{graph_id}"
        
        # 创建子图
        success = neo4j_service.create_subgraph(subgraph_name)
        
        if success:
            # 更新图谱信息
            graph.neo4j_subgraph = subgraph_name
            graph.neo4j_status = "created"
            
            # 创建向量索引
            neo4j_service.create_vector_index(subgraph_name)
            
            # 获取统计信息
            stats = neo4j_service.get_subgraph_statistics(subgraph_name)
            graph.neo4j_stats = stats
            
            # 保存到数据库
            db.add(graph)
            db.commit()
            
            return {
                "status": "success",
                "message": "Neo4j子图创建成功",
                "data": {
                    "subgraph_name": subgraph_name,
                    "neo4j_status": "created",
                    "statistics": stats
                }
            }
        else:
            return {
                "status": "error",
                "message": "Neo4j子图创建失败",
                "data": {
                    "subgraph_name": None,
                    "neo4j_status": "error"
                }
            }
    except Exception as e:
        graph.neo4j_status = "error"
        db.add(graph)
        db.commit()
        
        return {
            "status": "error",
            "message": f"Neo4j子图创建失败: {str(e)}",
            "data": {
                "error": str(e)
            }
        }

@router.delete("/{graph_id}/neo4j-subgraph")
async def delete_neo4j_subgraph(
    graph_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    删除Neo4j子图
    """
    # 获取图谱信息
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查是否有子图
    if not graph.neo4j_subgraph:
        return {
            "status": "warning",
            "message": "该知识图谱没有关联的Neo4j子图",
            "data": {
                "neo4j_status": graph.neo4j_status
            }
        }
    
    try:
        # 获取Neo4j配置
        neo4j_config = get_neo4j_config()
        
        # 创建Neo4j服务
        neo4j_service = get_neo4j_service(
            uri=neo4j_config["uri"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            database=neo4j_config["database"],
            force_new=True
        )
        
        # 删除子图
        success = neo4j_service.delete_subgraph(graph.neo4j_subgraph)
        
        if success:
            # 更新图谱信息
            graph.neo4j_subgraph = None
            graph.neo4j_status = "pending"
            graph.neo4j_stats = None
            
            # 保存到数据库
            db.add(graph)
            db.commit()
            
            return {
                "status": "success",
                "message": "Neo4j子图删除成功",
                "data": {
                    "neo4j_status": "pending"
                }
            }
        else:
            return {
                "status": "error",
                "message": "Neo4j子图删除失败",
                "data": {
                    "neo4j_status": graph.neo4j_status
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Neo4j子图删除失败: {str(e)}",
            "data": {
                "error": str(e)
            }
        }

@router.get("/{graph_id}/neo4j-visualization")
async def get_neo4j_visualization(
    graph_id: str,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    获取Neo4j子图可视化数据
    """
    # 获取图谱信息
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查是否有子图
    if not graph.neo4j_subgraph or graph.neo4j_status != "created":
        return {
            "code": 200,
            "data": {
                "nodes": [],
                "links": []
            },
            "msg": "该知识图谱没有关联的Neo4j子图或子图尚未创建完成"
        }
    
    try:
        # 获取Neo4j配置
        neo4j_config = get_neo4j_config()
        
        # 创建Neo4j服务
        neo4j_service = get_neo4j_service(
            uri=neo4j_config["uri"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            database=neo4j_config["database"],
            force_new=True
        )
        
        # 获取可视化数据
        data = neo4j_service.get_graph_visualization_data(graph.neo4j_subgraph, limit)
        
        # 检查数据是否为空
        message = "操作成功"
        if data and data.get("nodes") and len(data["nodes"]) == 0:
            message = "知识图谱中没有节点数据，请先抽取知识"
        
        return {
            "code": 200,
            "data": data,
            "msg": message
        }
    except Exception as e:
        logger.error(f"获取Neo4j可视化数据失败: {str(e)}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return {
            "code": 500,
            "data": {
                "nodes": [],
                "links": []
            },
            "msg": f"获取Neo4j可视化数据失败: {str(e)}"
        }

@router.post("/{graph_id}/extract-with-neo4j")
async def extract_with_neo4j(
    graph_id: str,
    background_tasks: BackgroundTasks,
    data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    使用Neo4j GraphRAG抽取知识
    """
    # 检查参数
    if "fileId" not in data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少必要参数: fileId"
        )
    
    file_id = data["fileId"]
    logger.info(f"开始Neo4j知识抽取: 图谱ID={graph_id}, 文件ID={file_id}")
    
    # 获取图谱信息
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查Neo4j子图状态
    if graph.neo4j_status != "created":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图谱的Neo4j子图未创建，请先创建Neo4j子图"
        )
    
    # 查询文件
    file = db.query(GraphFile).filter(GraphFile.id == file_id).first()
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="文件不存在"
        )
    
    # 检查文件是否已解析
    if file.status not in ["parsed", "processed", "extracted", "completed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件尚未解析完成"
        )
    
    # 创建任务记录
    task = ExtractionTask(
        graph_id=graph_id,
        file_id=file_id,
        task_type="neo4j_extraction",
        status="pending",
        parameters={"use_neo4j": True},
        result=None,
        model_id=graph.model_id
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # 启动异步任务
    background_tasks.add_task(
        extract_knowledge_with_neo4j_task,
        db_session_maker=db_session, 
        task_id=task.id,
        graph_id=graph_id,
        file_id=file_id
    )
    
    return {
        "success": True,
        "message": "Neo4j知识抽取任务已提交",
        "taskId": task.id
    }

@router.post("/{graph_id}/extract-with-llm")
async def extract_with_llm(
    graph_id: str,
    background_tasks: BackgroundTasks,
    data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    使用关联的大模型抽取知识并存储到Neo4j
    """
    # 检查参数
    if "fileId" not in data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少必要参数: fileId"
        )
    
    file_id = data["fileId"]
    logger.info(f"开始使用LLM进行知识抽取: 图谱ID={graph_id}, 文件ID={file_id}")
    
    # 获取图谱信息
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查Neo4j子图状态
    if graph.neo4j_status != "created":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图谱的Neo4j子图未创建，请先创建Neo4j子图"
        )
    
    # 查询文件
    file = db.query(GraphFile).filter(GraphFile.id == file_id).first()
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="文件不存在"
        )
    
    # 检查文件是否已解析
    if file.status not in ["parsed", "processed", "extracted", "completed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件尚未解析完成"
        )
    
    # 检查图谱是否有关联模型
    if not graph.model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该知识图谱未关联对话模型，请先在图谱设置中关联模型"
        )
    
    # 创建任务记录
    task = ExtractionTask(
        graph_id=graph_id,
        file_id=file_id,
        task_type="llm_extraction",
        status="pending",
        parameters={"use_llm": True},
        result=None,
        model_id=graph.model_id
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # 启动异步任务
    background_tasks.add_task(
        extract_knowledge_with_llm_task,
        db_session_maker=db_session, 
        task_id=task.id,
        graph_id=graph_id,
        file_id=file_id
    )
    
    return {
        "success": True,
        "message": "知识抽取任务已提交",
        "taskId": task.id
    } 

# 添加新的API端点 - 关联已有文件到知识图谱
@router.post("/{graph_id}/associate-files")
async def associate_files_to_graph(
    graph_id: str,
    data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    将已有的文件关联到知识图谱
    
    参数:
        graph_id: 知识图谱ID
        data: 包含文件ID列表的数据对象，格式为 {"file_ids": [id1, id2, ...]}
    
    返回:
        关联结果
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查参数
    if "file_ids" not in data or not isinstance(data["file_ids"], list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少必要参数: file_ids，应为文件ID的列表"
        )
    
    file_ids = data["file_ids"]
    if not file_ids:
        return {
            "code": status.HTTP_400_BAD_REQUEST,
            "message": "文件ID列表为空",
            "data": {"associated_count": 0}
        }
    
    # 获取要关联的文件
    from app.models.file import File as FileModel
    
    existing_files = db.query(FileModel).filter(FileModel.id.in_(file_ids)).all()
    if not existing_files:
        return {
            "code": status.HTTP_404_NOT_FOUND,
            "message": "未找到指定的文件",
            "data": {"associated_count": 0}
        }
    
    # 检查哪些文件已经关联到该图谱
    from app.models.graph_file import GraphFile
    
    existing_graph_files = db.query(GraphFile).filter(
        GraphFile.graph_id == graph_id,
        GraphFile.original_file_id.in_(file_ids)
    ).all()
    
    existing_file_ids = [gf.original_file_id for gf in existing_graph_files]
    
    # 处理每个文件
    associated_files = []
    for file in existing_files:
        # 跳过已关联的文件
        if file.id in existing_file_ids:
            continue
        
        try:
            # 创建GraphFile记录，关联文件到图谱
            graph_file = GraphFile(
                graph_id=graph_id,
                filename=file.filename,
                original_filename=file.original_filename,
                file_type=file.file_type,
                file_size=file.file_size,
                original_file_id=file.id,  # 记录原始文件ID
                path=file.path,  # 使用原始文件路径
                status=file.status,  # 继承原始文件状态
                text_content=file.text_content,  # 复制文本内容
                created_by=current_user.id
            )
            
            db.add(graph_file)
            db.commit()
            db.refresh(graph_file)
            
            associated_files.append(graph_file.to_dict())
        except Exception as e:
            # 单个文件关联失败不应该影响整个批处理
            print(f"关联文件 {file.id} 失败: {str(e)}")
    
    return {
        "code": status.HTTP_200_OK,
        "message": f"成功关联 {len(associated_files)} 个文件到知识图谱",
        "data": {
            "files": associated_files,
            "associated_count": len(associated_files)
        }
    }

# 将extract_knowledge接口修改为支持操作文件管理中的文件
@router.post("/{graph_id}/extract-from-file")
async def extract_from_file(
    graph_id: str,
    background_tasks: BackgroundTasks,
    data: dict = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_active_user),
):
    """
    从文件管理中的文件提取知识并添加到图谱
    
    参数:
        graph_id: 知识图谱ID
        data: 包含文件ID的数据对象，格式为 {"file_id": "file_id", "modelId": "model_id"}
    
    返回:
        任务创建结果
    """
    # 检查知识图谱是否存在
    graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识图谱不存在"
        )
    
    # 检查参数
    if "file_id" not in data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少必要参数: file_id"
        )
    
    file_id = data["file_id"]
    
    # 检查图谱是否关联了模型
    model_id = data.get("modelId") or graph.model_id
    if not model_id:
        return {
            "code": status.HTTP_400_BAD_REQUEST,
            "message": "知识图谱未关联模型，请先在基本信息中关联一个对话模型或在请求中提供modelId"
        }
    
    # 查询文件
    from app.models.file import File as FileModel
    
    file = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在"
        )
    
    # 检查文件状态：需要是已处理的文件
    if file.status not in ["processed", "completed"]:
        return {
            "code": status.HTTP_400_BAD_REQUEST,
            "message": f"文件尚未处理完成，当前状态: {file.status}",
            "data": {"file_status": file.status}
        }
    
    # 检查Neo4j子图状态
    if graph.neo4j_status != "created":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图谱的Neo4j子图未创建，请先创建Neo4j子图"
        )
    
    # 将文件关联到图谱（如果还未关联）
    from app.models.graph_file import GraphFile
    
    graph_file = db.query(GraphFile).filter(
        GraphFile.graph_id == graph_id,
        GraphFile.original_file_id == file_id
    ).first()
    
    if not graph_file:
        # 创建新的关联
        graph_file = GraphFile(
            graph_id=graph_id,
            filename=file.filename,
            original_filename=file.original_filename,
            file_type=file.file_type,
            file_size=file.file_size,
            original_file_id=file.id,
            path=file.path,
            status=file.status,
            text_content=file.text_content,
            created_by=current_user.id
        )
        
        db.add(graph_file)
        db.commit()
        db.refresh(graph_file)
    
    # 创建抽取任务
    from app.models.extraction_task import ExtractionTask
    
    task = ExtractionTask(
        graph_id=graph_id,
        file_id=graph_file.id,  # 使用关联后的文件ID
        original_file_id=file_id,  # 记录原始文件ID
        task_type="llm_extraction",
        status="pending",
        parameters={"use_llm": True, "dynamic_schema": graph.dynamic_schema},
        result=None,
        model_id=model_id
    )
    
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # 启动异步任务
    background_tasks.add_task(
        extract_knowledge_with_llm_task,
        db_session_maker=db_session, 
        task_id=task.id,
        graph_id=graph_id,
        file_id=graph_file.id
    )
    
    return {
        "code": status.HTTP_200_OK,
        "message": "知识抽取任务已提交",
        "data": {
            "taskId": task.id,
            "fileId": graph_file.id,
            "originalFileId": file_id,
            "status": "pending"
        }
    } 