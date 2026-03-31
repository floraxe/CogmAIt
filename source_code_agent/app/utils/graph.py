from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import uuid
import asyncio
import traceback
import json
import time
import logging
import tempfile
import subprocess
import urllib.parse
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi import HTTPException, UploadFile, BackgroundTasks

from app.models.graph import Graph, GraphNode, GraphEdge
from app.models.graph_file import GraphFile
from app.models.extraction_task import ExtractionTask
from app.models.graph import Graph as GraphModel
from app.schemas.graph import (
    GraphCreate, 
    GraphUpdate, 
    GraphNodeCreate,
    GraphNodeUpdate,
    GraphEdgeCreate,
    GraphEdgeUpdate
)
from app.utils.model import get_model, execute_model_inference
from app.utils.file_processor import process_file
from app.db.session import get_db

from datetime import datetime
from app.utils.neo4j_graphrag import get_neo4j_graphrag
from app.utils.neo4j_utils import get_neo4j_service
from app.utils.config import get_neo4j_config
# 配置日志
logger = logging.getLogger(__name__)

def get_graph(db: Session, graph_id: str) -> Optional[Graph]:
    """
    通过ID获取知识图谱
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 图谱ID
    
    返回:
        Optional[Graph]: 图谱对象或None
    """
    return db.query(Graph).filter(Graph.id == graph_id).first()


def get_graph_list(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    name: Optional[str] = None,
    status: Optional[str] = None
) -> List[Graph]:
    """
    获取知识图谱列表，支持过滤
    
    参数:
        db (Session): 数据库会话
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        name (Optional[str]): 按名称过滤
        status (Optional[str]): 按状态过滤
    
    返回:
        List[Graph]: 图谱列表
    """
    query = db.query(Graph)
    
    # 应用过滤条件
    if name:
        query = query.filter(Graph.name.ilike(f"%{name}%"))
    if status:
        query = query.filter(Graph.status == status)
    
    # 应用分页
    return query.offset(skip).limit(limit).all()


def create_graph(db: Session, graph_in: GraphCreate, user_id: str = None) -> Graph:
    """
    创建新知识图谱
    
    参数:
        db (Session): 数据库会话
        graph_in (GraphCreate): 图谱创建模式
        user_id (str): 创建者用户ID
    
    返回:
        Graph: 创建的图谱
    """
    # 创建新图谱对象
    db_graph = Graph(
        name=graph_in.name,
        description=graph_in.description,
        config=graph_in.config,
        user_id=user_id  # 添加用户ID
    )
    
    # 添加到数据库
    db.add(db_graph)
    db.commit()
    db.refresh(db_graph)
    
    return db_graph


def update_graph(db: Session, graph: Graph, graph_in: GraphUpdate) -> Graph:
    """
    更新知识图谱信息
    
    参数:
        db (Session): 数据库会话
        graph (Graph): 要更新的图谱
        graph_in (GraphUpdate): 图谱更新模式
    
    返回:
        Graph: 更新后的图谱
    """
    # 获取更新数据
    update_data = graph_in.dict(exclude_unset=True)
    
    # 更新模型属性
    for field, value in update_data.items():
        if hasattr(graph, field) and value is not None:
            setattr(graph, field, value)
    
    db.add(graph)
    db.commit()
    db.refresh(graph)
    
    return graph


def delete_graph(db: Session, graph_id: str) -> None:
    """
    删除知识图谱
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 图谱ID
    """
    graph = get_graph(db, graph_id)
    if graph:
        db.delete(graph)
        db.commit()


# 节点操作 #

def get_graph_node(db: Session, node_id: str) -> Optional[GraphNode]:
    """
    通过ID获取图谱节点
    
    参数:
        db (Session): 数据库会话
        node_id (str): 节点ID
    
    返回:
        Optional[GraphNode]: 节点对象或None
    """
    return db.query(GraphNode).filter(GraphNode.id == node_id).first()


def get_graph_nodes(
    db: Session,
    graph_id: str,
    skip: int = 0,
    limit: int = 100,
    node_type: Optional[str] = None,
    name: Optional[str] = None
) -> List[GraphNode]:
    """
    获取图谱的节点列表
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 图谱ID
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        node_type (Optional[str]): 按节点类型过滤
        name (Optional[str]): 按名称过滤
        
    返回:
        List[GraphNode]: 节点列表
    """
    query = db.query(GraphNode).filter(GraphNode.graph_id == graph_id)
    
    # 应用过滤条件
    if node_type:
        query = query.filter(GraphNode.node_type == node_type)
    if name:
        query = query.filter(GraphNode.name.ilike(f"%{name}%"))
    
    # 应用分页和排序
    return query.order_by(GraphNode.created_at.desc()).offset(skip).limit(limit).all()


def create_graph_node(
    db: Session,
    graph_id: str,
    node_in: GraphNodeCreate
) -> GraphNode:
    """
    创建图谱节点
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 图谱ID
        node_in (GraphNodeCreate): 节点创建模式
        
    返回:
        GraphNode: 创建的节点
    """
    # 检查图谱是否存在
    graph = get_graph(db, graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail="知识图谱不存在")
    
    # 创建节点
    db_node = GraphNode(
        graph_id=graph_id,
        name=node_in.name,
        node_type=node_in.node_type,
        properties=node_in.properties,
        description=node_in.description
    )
    
    # 添加到数据库
    db.add(db_node)
    
    # 更新图谱节点计数
    graph.entity_count += 1
    db.add(graph)
    
    db.commit()
    db.refresh(db_node)
    
    return db_node


def update_graph_node(
    db: Session,
    node: GraphNode,
    node_in: GraphNodeUpdate
) -> GraphNode:
    """
    更新图谱节点
    
    参数:
        db (Session): 数据库会话
        node (GraphNode): 要更新的节点
        node_in (GraphNodeUpdate): 节点更新模式
        
    返回:
        GraphNode: 更新后的节点
    """
    # 获取更新数据
    update_data = node_in.dict(exclude_unset=True)
    
    # 更新节点属性
    for field, value in update_data.items():
        if hasattr(node, field) and value is not None:
            setattr(node, field, value)
    
    db.add(node)
    db.commit()
    db.refresh(node)
    
    return node


def delete_graph_node(db: Session, node_id: str) -> None:
    """
    删除图谱节点
    
    参数:
        db (Session): 数据库会话
        node_id (str): 节点ID
    """
    node = get_graph_node(db, node_id)
    if node:
        # 获取图谱
        graph = get_graph(db, node.graph_id)
        
        # 删除与该节点相关的边
        related_edges = (
            db.query(GraphEdge)
            .filter(
                or_(
                    GraphEdge.source_id == node_id,
                    GraphEdge.target_id == node_id
                )
            )
            .all()
        )
        
        for edge in related_edges:
            db.delete(edge)
            if graph:
                graph.relation_count = max(0, graph.relation_count - 1)
        
        # 删除节点
        db.delete(node)
        
        # 更新图谱节点计数
        if graph:
            graph.entity_count = max(0, graph.entity_count - 1)
            db.add(graph)
        
        db.commit()


# 边操作 #

def get_graph_edge(db: Session, edge_id: str) -> Optional[GraphEdge]:
    """
    通过ID获取图谱边
    
    参数:
        db (Session): 数据库会话
        edge_id (str): 边ID
    
    返回:
        Optional[GraphEdge]: 边对象或None
    """
    return db.query(GraphEdge).filter(GraphEdge.id == edge_id).first()


def get_graph_edges(
    db: Session,
    graph_id: str,
    skip: int = 0,
    limit: int = 100,
    relation_type: Optional[str] = None,
    source_id: Optional[str] = None,
    target_id: Optional[str] = None
) -> List[GraphEdge]:
    """
    获取图谱的边列表
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 图谱ID
        skip (int): 跳过的记录数
        limit (int): 限制返回的记录数
        relation_type (Optional[str]): 按关系类型过滤
        source_id (Optional[str]): 按源节点过滤
        target_id (Optional[str]): 按目标节点过滤
        
    返回:
        List[GraphEdge]: 边列表
    """
    query = db.query(GraphEdge).filter(GraphEdge.graph_id == graph_id)
    
    # 应用过滤条件
    if relation_type:
        query = query.filter(GraphEdge.relation_type == relation_type)
    if source_id:
        query = query.filter(GraphEdge.source_id == source_id)
    if target_id:
        query = query.filter(GraphEdge.target_id == target_id)
    
    # 应用分页和排序
    return query.order_by(GraphEdge.created_at.desc()).offset(skip).limit(limit).all()


def create_graph_edge(
    db: Session,
    graph_id: str,
    edge_in: GraphEdgeCreate
) -> GraphEdge:
    """
    创建图谱边
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 图谱ID
        edge_in (GraphEdgeCreate): 边创建模式
        
    返回:
        GraphEdge: 创建的边
    """
    # 检查图谱是否存在
    graph = get_graph(db, graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail="知识图谱不存在")
    
    # 检查源节点是否存在
    source_node = get_graph_node(db, edge_in.source_id)
    if not source_node or source_node.graph_id != graph_id:
        raise HTTPException(status_code=404, detail="源节点不存在或不属于该图谱")
    
    # 检查目标节点是否存在
    target_node = get_graph_node(db, edge_in.target_id)
    if not target_node or target_node.graph_id != graph_id:
        raise HTTPException(status_code=404, detail="目标节点不存在或不属于该图谱")
    
    # 创建边
    db_edge = GraphEdge(
        graph_id=graph_id,
        source_id=edge_in.source_id,
        target_id=edge_in.target_id,
        relation_type=edge_in.relation_type,
        properties=edge_in.properties,
        weight=edge_in.weight,
        bidirectional=edge_in.bidirectional
    )
    
    # 添加到数据库
    db.add(db_edge)
    
    # 更新图谱边计数
    graph.relation_count += 1
    db.add(graph)
    
    db.commit()
    db.refresh(db_edge)
    
    return db_edge


def update_graph_edge(
    db: Session,
    edge: GraphEdge,
    edge_in: GraphEdgeUpdate
) -> GraphEdge:
    """
    更新图谱边
    
    参数:
        db (Session): 数据库会话
        edge (GraphEdge): 要更新的边
        edge_in (GraphEdgeUpdate): 边更新模式
        
    返回:
        GraphEdge: 更新后的边
    """
    # 获取更新数据
    update_data = edge_in.dict(exclude_unset=True)
    
    # 更新边属性
    for field, value in update_data.items():
        if hasattr(edge, field) and value is not None:
            setattr(edge, field, value)
    
    db.add(edge)
    db.commit()
    db.refresh(edge)
    
    return edge


def delete_graph_edge(db: Session, edge_id: str) -> None:
    """
    删除图谱边
    
    参数:
        db (Session): 数据库会话
        edge_id (str): 边ID
    """
    edge = get_graph_edge(db, edge_id)
    if edge:
        # 获取图谱
        graph = get_graph(db, edge.graph_id)
        
        # 删除边
        db.delete(edge)
        
        # 更新图谱边计数
        if graph:
            graph.relation_count = max(0, graph.relation_count - 1)
            db.add(graph)
        
        db.commit()


def get_graph_visualization_data(db: Session, graph_id: str) -> Dict[str, Any]:
    """
    获取图谱可视化数据
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 图谱ID
        
    返回:
        Dict[str, Any]: 包含节点和边的可视化数据
    """
    # 获取图谱节点
    nodes = db.query(GraphNode).filter(GraphNode.graph_id == graph_id).all()
    
    # 获取图谱边
    edges = db.query(GraphEdge).filter(GraphEdge.graph_id == graph_id).all()
    
    # 转换为可视化格式
    visualization_nodes = []
    for node in nodes:
        visualization_nodes.append({
            "id": node.id,
            "name": node.name,
            "type": node.node_type,
            "properties": node.properties or {}
        })
    
    visualization_edges = []
    for edge in edges:
        visualization_edges.append({
            "id": edge.id,
            "source": edge.source_id,
            "target": edge.target_id,
            "type": edge.relation_type,
            "weight": edge.weight,
            "bidirectional": edge.bidirectional,
            "properties": edge.properties or {}
        })
    
    return {
        "nodes": visualization_nodes,
        "edges": visualization_edges
    }


async def upload_file_to_graph(
    db: Session, 
    graph_id: str, 
    file: UploadFile,
    background_tasks: BackgroundTasks,
    upload_dir: str = "uploads",
    user_id: str = None
) -> GraphFile:
    """
    上传文件到知识图谱，按照新的处理流程异步处理文件
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 知识图谱ID
        file (UploadFile): 上传的文件
        background_tasks (BackgroundTasks): 后台任务对象，用于异步处理文件
        upload_dir (str): 上传目录
        user_id (str): 创建者用户ID
    
    返回:
        GraphFile: 创建的文件记录
    """
    # 检查知识图谱是否存在
    graph = get_graph(db, graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail="知识图谱不存在")
    
    # 获取文件信息
    original_filename = file.filename
    file_extension = original_filename.split('.')[-1] if '.' in original_filename else ''
    file_type = file_extension.lower()
    
    # 生成唯一文件名
    filename_uuid = uuid.uuid4().hex
    unique_filename = f"{filename_uuid}.{file_extension}"
    
    # 创建上传目录
    graph_dir = os.path.join(upload_dir, "graph", graph_id)
    os.makedirs(graph_dir, exist_ok=True)
    
    # 文件路径
    file_path = os.path.join(graph_dir, unique_filename)
    
    # 创建文件记录，初始状态为上传中
    db_file = GraphFile(
        graph_id=graph_id,
        filename=unique_filename,
        original_filename=original_filename,
        file_type=file_type,
        path=file_path,
        status="uploading",  # 设置初始状态为上传中
        user_id=user_id  # 添加用户ID
    )
    
    # 添加到数据库
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    
    try:
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            file_size = len(content)
        
        # 更新文件大小和状态为上传完成
        db_file.file_size = file_size
        db_file.status = "uploaded"  # 上传完成
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        # 添加后台任务处理文件内容提取
        background_tasks.add_task(
            process_graph_file,
            db_file.id,
            graph_id
        )
        
        return db_file

    except Exception as e:
        # 如果上传过程中出错，更新状态为失败
        db_file.status = "failed"
        db_file.error = f"文件上传失败: {str(e)}"
        db.add(db_file)
        db.commit()
        
        # 重新抛出异常，让上层处理
        raise e

def delete_graph_file(db: Session, file_id: str) -> None:
    """
    删除知识图谱文件
    
    参数:
        db (Session): 数据库会话
        file_id (str): 文件ID
    """
    file = db.query(GraphFile).filter(GraphFile.id == file_id).first()
    if file:
        # 从文件系统删除文件
        if os.path.exists(file.path):
            try:
                os.remove(file.path)
            except Exception as e:
                print(f"删除文件失败: {str(e)}")
        
        # 从数据库删除记录
        db.delete(file)
        db.commit()

def get_graph_file(db: Session, file_id: str) -> Optional[GraphFile]:
    """
    获取知识图谱文件
    
    参数:
        db (Session): 数据库会话
        file_id (str): 文件ID
    
    返回:
        Optional[GraphFile]: 文件对象或None
    """
    return db.query(GraphFile).filter(GraphFile.id == file_id).first()

async def process_graph_file(file_id: str, graph_id: str):
    """
    处理知识图谱文件（异步后台任务）
    
    参数:
        file_id (str): 文件ID
        graph_id (str): 知识图谱ID
    """
    from app.db.session import SessionLocal
    import traceback
    import os
    from app.utils.file_processor import process_file
    
    db = SessionLocal()
    try:
        # 获取文件
        file = get_graph_file(db, file_id)
        if not file:
            print(f"文件不存在: {file_id}")
            return
        
        # 更新文件状态为处理中
        file.status = "processing"
        db.add(file)
        db.commit()
        
        # 根据文件类型处理文件内容
        file_path = file.path
        file_type = file.file_type.lower()
        
        # 创建输出目录
        output_dir = os.path.join(os.getcwd(), "uploads", "processed", graph_id, file.filename)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 调用 file_processor 中的 process_file 函数处理文件
            result = process_file(
                file_path=file_path, 
                output_dir=output_dir,
                filename_uuid=file.filename,
                knowledge_id=graph_id
            )
            
            if "error" in result:
                # 处理失败
                file.status = "failed"
                file.error = result["error"]
                db.add(file)
                db.commit()
                print(f"文件 {file_id} 处理失败: {result['error']}")
                return
            
            # 读取markdown文件内容
            md_path = result.get("markdown_path")
            if md_path and os.path.exists(md_path):
                with open(md_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                    
                # 将提取的文本内容保存到数据库
                file.text_content = text_content
                file.status = "parsed"
                file.extra_data = result  # 保存处理结果的元数据
                db.add(file)
                db.commit()
                
                print(f"文件 {file_id} 解析完成，文本内容已保存到数据库")
            else:
                # 如果没有生成markdown文件，尝试直接从文件中提取文本
                text_content = await extract_text_from_file(file_path, file_type)
                
                # 更新文件内容
                file.text_content = text_content
                file.status = "parsed"
                file.extra_data = result
                db.add(file)
                db.commit()
                
                print(f"文件 {file_id} 解析完成 (使用直接提取方法)")
        except Exception as e:
            # 更新文件状态为失败
            file.status = "failed"
            file.error = f"文件解析失败: {str(e)}"
            db.add(file)
            db.commit()
            
            traceback.print_exc()
            print(f"文件 {file_id} 解析失败: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        print(f"处理文件时出错: {str(e)}")
    finally:
        db.close()

async def extract_text_from_file(file_path: str, file_type: str) -> str:
    """
    从文件中提取文本内容
    
    参数:
        file_path (str): 文件路径
        file_type (str): 文件类型
    
    返回:
        str: 提取的文本内容
    """
    try:
        # 这里应该根据文件类型调用不同的文本提取库
        # 例如PDF可以使用PyPDF2或pdfplumber，Word可以使用python-docx等
        
        # 这里仅作为示例，实际应用中需要安装相应的库
        if file_type == 'pdf':
            # 模拟PDF文本提取
            return await simulate_text_extraction(file_path, "PDF")
        elif file_type in ['doc', 'docx']:
            # 模拟Word文本提取
            return await simulate_text_extraction(file_path, "Word")
        elif file_type in ['txt', 'md']:
            # 读取文本文件
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"不支持的文件类型: {file_type}"
    except Exception as e:
        print(f"文本提取失败: {str(e)}")
        traceback.print_exc()
        return f"文本提取失败: {str(e)}"

async def simulate_text_extraction(file_path: str, file_type: str) -> str:
    """
    模拟文本提取过程
    
    参数:
        file_path (str): 文件路径
        file_type (str): 文件类型
    
    返回:
        str: 模拟提取的文本内容
    """
    # 实际应用中，这里应该调用相应的文本提取库
    # 这里仅作为示例
    await asyncio.sleep(1)  # 模拟处理时间
    
    return f"这是从{file_type}文件 {os.path.basename(file_path)} 中提取的文本内容示例。实际应用中，这里应该是提取的真实内容。"

async def extract_knowledge_from_file(
    task_id: str,
    graph_id: str,
    file_id: str,
    model_id: Optional[str] = None,
    prompt: str = "",
    parameters: Dict[str, Any] = None
):
    """
    从文件内容中提取知识图谱数据（异步后台任务）
    
    参数:
        task_id (str): 任务ID
        graph_id (str): 知识图谱ID
        file_id (str): 文件ID
        model_id (Optional[str]): 模型ID，可选
        prompt (str): 提示词
        parameters (Dict[str, Any]): 参数
    """
    from app.db.session import SessionLocal
    import time
    import traceback
    from app.utils.model import get_model, execute_model_inference
    from app.models.graph import GraphNode, GraphEdge
    
    # 确保parameters是字典
    if parameters is None:
        parameters = {}
    
    # 检查是否使用Neo4j-GraphRAG
    use_neo4j = parameters.get("use_neo4j", False)
    
    db = SessionLocal()
    try:
        # 获取任务
        task = db.query(ExtractionTask).filter(ExtractionTask.id == task_id).first()
        if not task:
            print(f"任务不存在: {task_id}")
            return
        
        # 更新任务状态
        task.status = "processing"
        task.start_time = datetime.now()
        db.add(task)
        db.commit()
        
        # 获取文件
        file = get_graph_file(db, file_id)
        if not file:
            task.status = "failed"
            task.error = f"文件不存在: {file_id}"
            db.add(task)
            db.commit()
            return
        
        # 获取图谱
        graph = get_graph(db, graph_id)
        if not graph:
            task.status = "failed"
            task.error = f"知识图谱不存在: {graph_id}"
            db.add(task)
            db.commit()
            return
        
        # 如果图谱已配置Neo4j子图且未在parameters中明确指定不使用Neo4j
        if graph.neo4j_subgraph and graph.neo4j_status == "created" and "use_neo4j" not in parameters:
            use_neo4j = True
        
        # 获取模型
        if not model_id and graph.model_id:
            model_id = graph.model_id
        
        if not model_id and not use_neo4j:
            task.status = "failed"
            task.error = "未指定模型，且知识图谱未关联默认模型"
            db.add(task)
            db.commit()
            return
        
        # 检查文件状态，确保文件已被解析并且有文本内容
        if file.status not in ["parsed", "completed", "extracted"]:
            # 如果文件未解析完成，等待一段时间
            max_wait_time = 60  # 最多等待60秒
            wait_start = time.time()
            
            while file.status not in ["parsed", "completed", "extracted"] and time.time() - wait_start < max_wait_time:
                # 更新文件状态
                file = get_graph_file(db, file_id)
                if file.status == "failed":
                    task.status = "failed"
                    task.error = f"文件解析失败: {file.error}"
                    db.add(task)
                    db.commit()
                    return
                    
                print(f"文件 {file_id} 当前状态为 {file.status}，等待解析完成...")
                time.sleep(3)  # 每3秒检查一次
                
            # 再次检查文件状态
            file = get_graph_file(db, file_id)
            if file.status not in ["parsed", "completed", "extracted"]:
                task.status = "failed"
                task.error = f"等待文件解析超时，当前状态: {file.status}"
                db.add(task)
                db.commit()
                return
        
        # 检查是否有文本内容
        if not file.text_content:
            task.status = "failed"
            task.error = "文件没有可用的文本内容"
            db.add(task)
            db.commit()
            return
        
        # 更新文件状态
        file.status = "extracting"
        db.add(file)
        db.commit()
        
        # 更新任务状态
        task.status = "running"
        db.add(task)
        db.commit()
        
        # 获取文本内容
        text_content = file.text_content
        
        # 如果使用Neo4j-GraphRAG
        if use_neo4j:
            # 调用Neo4j知识提取函数
            neo4j_result = extract_knowledge_with_neo4j(db, task, graph, file)
            
            if neo4j_result.get("success", False):
                # 提取成功，更新图谱统计信息
                update_graph_statistics(db, graph)
                
                # 获取最新的统计信息
                stats = graph.neo4j_stats or {}
                
                # 更新结果中的实体和关系计数
                if "entityCount" not in neo4j_result and stats:
                    neo4j_result["entityCount"] = stats.get("nodeCount", 0)
                    
                if "relationCount" not in neo4j_result and stats:
                    neo4j_result["relationCount"] = stats.get("edgeCount", 0)
                
                # 更新任务状态
                task.status = "completed"
                task.result = neo4j_result
                task.message = "使用Neo4j-GraphRAG知识提取完成"
                task.end_time = datetime.now()
                
                # 更新任务的实体和关系计数
                task.entity_count = neo4j_result.get("entityCount", 0)
                task.relation_count = neo4j_result.get("relationCount", 0)
                
                # 更新文件状态
                file.status = "extracted"
                
                # 保存更改
                db.add(task)
                db.add(file)
                db.commit()
                
                print(f"Neo4j-GraphRAG知识提取完成: {task_id}")
                return
            else:
                # 提取失败
                task.status = "failed"
                task.error = neo4j_result.get("error", "Neo4j-GraphRAG知识提取失败")
                task.message = "Neo4j-GraphRAG知识提取失败"
                task.end_time = datetime.now()
                db.add(task)
                db.commit()
                
                print(f"Neo4j-GraphRAG知识提取失败: {task_id} - {neo4j_result.get('error')}")
                return
        else:
            # 构建知识抽取的提示词
            if not prompt:
                # 使用默认提示词
                schema_data = {}
                if graph.config and "schema" in graph.config:
                    schema_data = graph.config["schema"]
                
                entity_types = []
                relation_types = []
                
                if "entityTypes" in schema_data:
                    entity_types = [{
                        "name": et.get("name", ""), 
                        "description": et.get("description", "")
                    } for et in schema_data["entityTypes"]]
                
                if "relationTypes" in schema_data:
                    relation_types = [{
                        "name": rt.get("name", ""),
                        "sourceType": rt.get("sourceType", ""),
                        "targetType": rt.get("targetType", ""),
                        "description": rt.get("description", "")
                    } for rt in schema_data["relationTypes"]]
                
                # 构建默认提示词
                prompt = f"""请从以下文本中提取知识图谱的实体和关系，并以JSON格式返回。

文本内容:
{text_content[:3000]}...

请按照以下JSON格式返回结果:
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "实体类型",
      "properties": {{
        "属性名1": "属性值1",
        "属性名2": "属性值2"
      }},
      "description": "实体描述"
    }}
  ],
  "relations": [
    {{
      "source": "源实体名称",
      "target": "目标实体名称",
      "type": "关系类型",
      "properties": {{
        "属性名1": "属性值1"
      }}
    }}
  ]
}}

"""
                # 如果有schema信息，添加到提示词中
                if entity_types:
                    prompt += f"\n可用的实体类型有: {[et['name'] for et in entity_types]}\n"
                    for et in entity_types:
                        if et.get("description"):
                            prompt += f"- {et['name']}: {et['description']}\n"
                
                if relation_types:
                    prompt += f"\n可用的关系类型有: {[rt['name'] for rt in relation_types]}\n"
                    for rt in relation_types:
                        source = rt.get("sourceType", "实体")
                        target = rt.get("targetType", "实体")
                        desc = rt.get("description", "")
                        prompt += f"- {rt['name']}: 从 {source} 到 {target}{' - ' + desc if desc else ''}\n"
            
            # 保存实际使用的提示词
            task.prompt = prompt
            db.add(task)
            db.commit()
            
            try:
                # 获取模型配置
                model = get_model(db, model_id)
                if not model:
                    raise ValueError(f"模型不存在: {model_id}")
                
                # 调用模型进行知识抽取
                response = await execute_model_inference(
                    db,
                    model_id,
                    {
                        "model_type": "chat",
                        "messages": [{"role": "user", "content": prompt}],
                    # parameters=parameters
                    }
                )
                print(response)
                # 检查响应
                if not response or "choices" not in response:
                    raise ValueError("模型响应无效")
                
                # 获取模型生成的文本
                generated_text = response["choices"][0]["message"]["content"]
                
                # 提取JSON部分
                json_str = extract_json_from_text(generated_text)
                if not json_str:
                    raise ValueError("未能从模型响应中提取JSON数据")
                
                # 解析JSON
                import json
                extraction_result = json.loads(json_str)
                
                # 检查JSON结构
                if "entities" not in extraction_result or "relations" not in extraction_result:
                    raise ValueError("JSON数据结构不符合要求，缺少entities或relations字段")
                
                # 保存结果到任务中
                task.result = extraction_result
                task.entity_count = len(extraction_result.get("entities", []))
                task.relation_count = len(extraction_result.get("relations", []))
                task.status = "completed"
                task.end_time = datetime.now()
                db.add(task)
                db.commit()
                
                # 保存实体和关系到知识图谱
                entities = extraction_result.get("entities", [])
                relations = extraction_result.get("relations", [])
                
                entity_count = 0
                relation_count = 0
                
                # 保存实体
                for entity in entities:
                    # 创建实体节点
                    node = GraphNode(
                        graph_id=graph_id,
                        name=entity.get('name', ''),
                        node_type=entity.get('type', ''),
                        properties=entity.get('properties', {}),
                        description=entity.get('description', '')
                    )
                    
                    db.add(node)
                    entity_count += 1
                
                db.commit()
                
                # 获取所有实体节点
                nodes = db.query(GraphNode).filter(GraphNode.graph_id == graph_id).all()
                node_map = {node.name: node for node in nodes}
                
                # 保存关系
                for relation in relations:
                    source_name = relation.get('source', '')
                    target_name = relation.get('target', '')
                    
                    # 查找源节点和目标节点
                    source_node = node_map.get(source_name)
                    target_node = node_map.get(target_name)
                    
                    if source_node and target_node:
                        # 创建关系边
                        edge = GraphEdge(
                            graph_id=graph_id,
                            source_id=source_node.id,
                            target_id=target_node.id,
                            relation=relation.get('type', ''),
                            properties=relation.get('properties', {})
                        )
                        
                        db.add(edge)
                        relation_count += 1
                
                db.commit()
                
                # 更新文件状态
                file.status = "extracted"
                db.add(file)
                
                # 更新图谱实体和关系计数
                graph.entity_count += entity_count
                graph.relation_count += relation_count
                db.add(graph)
                
                # 如果图谱有Neo4j子图，更新统计信息
                if graph.neo4j_subgraph:
                    update_graph_statistics(db, graph)
                
                # 更新任务状态
                task.status = "completed"
                task.message = f"已成功提取 {entity_count} 个实体和 {relation_count} 个关系"
                db.add(task)
                db.commit()
                
                print(f"知识抽取完成: 文件 {file_id}, 实体 {entity_count}, 关系 {relation_count}")
                
            except Exception as e:
                traceback.print_exc()
                # 更新任务状态为失败
                task.status = "failed"
                task.error = f"知识抽取失败: {str(e)}"
                task.end_time = datetime.now()
                db.add(task)
                
                # 恢复文件状态
                file.status = "parsed"  # 文件本身解析成功，只是知识抽取失败
                db.add(file)
                
                db.commit()
                print(f"知识抽取失败: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        print(f"知识抽取任务执行出错: {str(e)}")
        try:
            # 尝试更新任务状态
            task = db.query(ExtractionTask).filter(ExtractionTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error = f"任务执行出错: {str(e)}"
                task.end_time = datetime.now()
                db.add(task)
        except:
            pass
    finally:
        db.close()

def extract_json_from_text(text):
    """
    从文本中提取JSON字符串
    """
    import re
    
    # 查找JSON对象模式
    json_pattern = r'```json\s*([\s\S]*?)\s*```|```([\s\S]*?)\s*```|\{[\s\S]*\}'
    
    match = re.search(json_pattern, text)
    if match:
        # 提取JSON部分
        json_str = match.group(1) or match.group(2) or match.group(0)
        
        # 清理JSON字符串
        json_str = json_str.strip()
        
        # 确保它以{开头、}结尾
        if not json_str.startswith('{'):
            json_str = json_str[json_str.find('{'):]
        if not json_str.endswith('}'):
            json_str = json_str[:json_str.rfind('}')+1]
            
        return json_str
    
    return None

def generate_schema_extraction_prompt(db, graph_id, text_content):
    """
    根据图谱的Schema生成知识抽取提示词
    """
    # 获取图谱信息
    graph = get_graph(db, graph_id)
    if not graph or not graph.config or "schema" not in graph.config:
        # 如果没有Schema定义，返回一个通用提示词
        return f'请根据文本内容，抽取相关的实体和关系，生成知识图谱数据。\n\n' + \
               f'下面是需要抽取知识的文本内容:\n\n{text_content[:5000]}...(文本内容太长已省略)\n\n' + \
               f'\n请按照以下JSON格式返回抽取结果：\n' + \
               '{\n' + \
               '  "entities": [\n' + \
               '    {"type": "实体类型名称", "name": "实体名称", "properties": {"属性名": "属性值"}}\n' + \
               '  ],\n' + \
               '  "relations": [\n' + \
               '    {"type": "关系类型名称", "source": "源实体名称", "target": "目标实体名称", "properties": {"属性名": "属性值"}}\n' + \
               '  ]\n' + \
               '}\n\n' + \
               '请直接返回有效的JSON格式数据，不要添加其他回复或解释。确保JSON格式正确可解析。'
    
    # 从图谱config中获取Schema数据
    schema = graph.config.get("schema", {})
    entity_types = schema.get("entityTypes", [])
    relation_types = schema.get("relationTypes", [])
    properties = schema.get("properties", [])
    
    # 构建提示词
    prompt = '请根据以下Schema定义和提供的文本内容，抽取相关的实体和关系，生成知识图谱数据。\n\n'
    
    # 添加实体类型信息
    if entity_types:
        prompt += '实体类型：\n'
        for entity in entity_types:
            prompt += f"- {entity.get('name', '未知')}："
            if entity.get('description'):
                prompt += entity.get('description')
            prompt += '\n'
            
            # 添加实体必需属性
            required_props = []
            required_prop_ids = entity.get('requiredProperties', [])
            if required_prop_ids:
                for prop_id in required_prop_ids:
                    prop = next((p for p in properties if str(p.get('id')) == str(prop_id)), None)
                    if prop:
                        required_props.append(prop.get('name', ''))
                
                if required_props:
                    prompt += f"  必需属性：{', '.join(required_props)}\n"
    
    # 添加关系类型信息
    if relation_types:
        prompt += '\n关系类型：\n'
        for relation in relation_types:
            source_type_id = relation.get('sourceType')
            target_type_id = relation.get('targetType')
            
            source_entity = next((e for e in entity_types if str(e.get('id')) == str(source_type_id)), None)
            target_entity = next((e for e in entity_types if str(e.get('id')) == str(target_type_id)), None)
            
            source_name = source_entity.get('name', '未知') if source_entity else '未知'
            target_name = target_entity.get('name', '未知') if target_entity else '未知'
            
            prompt += f"- {relation.get('name', '未知')}：从{source_name}到{target_name}"
            if relation.get('description'):
                prompt += f"，{relation.get('description')}"
            prompt += '\n'
            
            # 添加关系必需属性
            required_props = []
            required_prop_ids = relation.get('requiredProperties', [])
            if required_prop_ids:
                for prop_id in required_prop_ids:
                    prop = next((p for p in properties if str(p.get('id')) == str(prop_id)), None)
                    if prop:
                        required_props.append(prop.get('name', ''))
                
                if required_props:
                    prompt += f"  必需属性：{', '.join(required_props)}\n"
    
    # 添加属性信息
    if properties:
        prompt += '\n属性定义：\n'
        for prop in properties:
            prompt += f"- {prop.get('name', '未知')}（{get_property_type_name(prop.get('type', ''))}）"
            if prop.get('description'):
                prompt += f"：{prop.get('description')}"
            prompt += '\n'
    
    # 添加文本内容
    prompt += f"\n下面是需要抽取知识的文本内容:\n\n{text_content[:5000]}...(文本内容太长已省略)\n\n"
    
    # 添加输出格式要求
    prompt += '\n请按照以下JSON格式返回抽取结果：\n'
    prompt += '{\n'
    prompt += '  "entities": [\n'
    prompt += '    {"type": "实体类型名称", "name": "实体名称", "properties": {"属性名": "属性值"}}\n'
    prompt += '  ],\n'
    prompt += '  "relations": [\n'
    prompt += '    {"type": "关系类型名称", "source": "源实体名称", "target": "目标实体名称", "properties": {"属性名": "属性值"}}\n'
    prompt += '  ]\n'
    prompt += '}\n'
    
    prompt += '\n请直接返回有效的JSON格式数据，不要添加其他回复或解释。确保JSON格式正确可解析。'
    
    return prompt

def get_property_type_name(type_value):
    """
    获取属性类型的可读名称
    """
    type_map = {
        'string': '字符串',
        'number': '数值',
        'integer': '整数',
        'boolean': '布尔值',
        'date': '日期',
        'datetime': '日期时间',
        'array': '数组',
        'object': '对象',
        'file': '文件',
        'image': '图片',
        'url': 'URL',
        'email': '邮箱',
        'phone': '电话',
        'id': '标识符'
    }
    return type_map.get(type_value, type_value) or '未知类型'

def extract_knowledge_with_neo4j(
    db: Session,
    task: ExtractionTask,
    graph: Graph,
    file: GraphFile
) -> Dict[str, Any]:
    """
    使用Neo4j-GraphRAG进行知识提取
    
    Args:
        db: 数据库会话
        task: 抽取任务
        graph: 知识图谱
        file: 文件
        
    Returns:
        Dict: 抽取结果
    """
    try:
        # 检查Neo4j子图是否已创建
        if not graph.neo4j_subgraph or graph.neo4j_status != "created":
            return {
                "success": False,
                "error": "知识图谱的Neo4j子图未创建，请先创建Neo4j子图"
            }
        
        # 获取Schema数据
        schema_data = None
        if graph.config and "schema" in graph.config:
            schema_data = graph.config["schema"]
        
        # 实例化GraphRAG工具
        neo4j_config = get_neo4j_config()
        graphrag = get_neo4j_graphrag(
            uri=neo4j_config["uri"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            database=neo4j_config["database"],
            force_new=True
        )
        
        # 检查任务参数中是否指定使用LLM
        use_llm = task.parameters.get("use_llm", False)
        
        # 使用Neo4j-GraphRAG提取知识
        result = graphrag.extract_knowledge_from_text(
            graph_id=graph.neo4j_subgraph,
            text=file.text_content,
            schema=schema_data,
            llm_config={
                "use_llm": use_llm,
                "db_session": db
            }
        )
        
        if result.get("success", False):
            # 提取成功，更新图谱统计信息
            update_graph_statistics(db, graph)
            
            # 获取最新的统计信息
            stats = graph.neo4j_stats or {}
            
            # 更新结果中的实体和关系计数
            if "entityCount" not in result and stats:
                result["entityCount"] = stats.get("nodeCount", 0)
                
            if "relationCount" not in result and stats:
                result["relationCount"] = stats.get("edgeCount", 0)
            
            return result
        else:
            return result
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Neo4j-GraphRAG知识提取异常: {str(e)}\n{error_traceback}")
        
        return {
            "success": False,
            "error": f"Neo4j-GraphRAG异常: {str(e)}"
        } 

async def extract_knowledge_with_neo4j_task(
    db_session_maker,
    task_id: str,
    graph_id: str,
    file_id: str
):
    """
    使用Neo4j-GraphRAG进行知识抽取的后台任务
    
    Args:
        db_session_maker: 数据库会话工厂
        task_id: 任务ID
        graph_id: 图谱ID
        file_id: 文件ID
    """
    
    db = db_session_maker()
    try:
        logger.info(f"开始Neo4j知识抽取任务: task_id={task_id}, graph_id={graph_id}, file_id={file_id}")
        
        # 获取任务
        task = db.query(ExtractionTask).filter(ExtractionTask.id == task_id).first()
        if not task:
            logger.error(f"未找到任务: {task_id}")
            return
        
        # 获取图谱
        graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
        if not graph:
            logger.error(f"未找到图谱: {graph_id}")
            task.status = "failed"
            task.error = "未找到图谱"
            task.end_time = datetime.now()
            db.add(task)
            db.commit()
            return
        
        # 获取文件
        file = db.query(GraphFile).filter(GraphFile.id == file_id).first()
        if not file or not file.text_content:
            logger.error(f"未找到文件或文件无文本内容: {file_id}")
            task.status = "failed"
            task.error = "未找到文件或文件无文本内容"
            task.end_time = datetime.now()
            db.add(task)
            db.commit()
            return
        
        # 更新任务状态
        task.status = "running"
        task.start_time = datetime.now()
        db.add(task)
        db.commit()
        
        # 标记文件为抽取中状态
        file.status = "extracting"
        db.add(file)
        db.commit()
        
        # 调用Neo4j知识抽取函数
        result = extract_knowledge_with_neo4j(
            db=db,
            task=task,
            graph=graph,
            file=file
        )
        
        logger.info(f"Neo4j知识抽取结果: success={result.get('success', False)}")
        
        # 处理结果
        if result.get("success", False):
            # 更新图谱统计信息
            update_graph_statistics(db, graph)
            
            # 更新文件状态
            file.status = "extracted"
            db.add(file)
            
            # 更新任务状态
            task.status = "completed"
            task.result = result
            task.end_time = datetime.now()
            
            # 更新任务的实体和关系计数
            if "entityCount" in result:
                task.entity_count = result["entityCount"]
            if "relationCount" in result:
                task.relation_count = result["relationCount"]
            
            db.add(task)
            db.commit()
            
            logger.info(f"Neo4j知识抽取成功: task_id={task.id}, 实体数={result.get('entityCount', 0)}, 关系数={result.get('relationCount', 0)}")
        else:
            # 更新任务状态为失败
            file.status = "extract_failed"
            db.add(file)
            
            task.status = "failed"
            task.error = result.get("error", "Neo4j知识抽取失败，未知错误")
            task.end_time = datetime.now()
            db.add(task)
            db.commit()
            
            logger.error(f"Neo4j知识抽取失败: task_id={task.id}, error={result.get('error')}")
    
    except Exception as e:
        logger.error(f"Neo4j知识抽取任务异常: {str(e)}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        
        # 尝试更新任务状态
        try:
            task = db.query(ExtractionTask).filter(ExtractionTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error = str(e)
                task.end_time = datetime.now()
                db.add(task)
                
                # 更新文件状态
                file = db.query(GraphFile).filter(GraphFile.id == file_id).first()
                if file:
                    file.status = "extract_failed"
                    db.add(file)
                
                db.commit()
        except Exception as e2:
            logger.error(f"更新任务状态失败: {str(e2)}")
    
    finally:
        db.close()

async def extract_knowledge_with_llm_task(
    db_session_maker,
    task_id: str,
    graph_id: str,
    file_id: str
):
    """
    使用关联大模型进行知识抽取的后台任务
    
    Args:
        db_session_maker: 数据库会话工厂
        task_id: 任务ID
        graph_id: 图谱ID
        file_id: 文件ID
    """
    from datetime import datetime
    from app.utils.llm_knowledge_extractor import LLMKnowledgeExtractor
    
    db = db_session_maker()
    try:
        logger.info(f"开始LLM知识抽取任务: task_id={task_id}, graph_id={graph_id}, file_id={file_id}")
        
        # 获取任务
        task = db.query(ExtractionTask).filter(ExtractionTask.id == task_id).first()
        if not task:
            logger.error(f"未找到任务: {task_id}")
            return
        
        # 获取图谱
        graph = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
        if not graph:
            logger.error(f"未找到图谱: {graph_id}")
            task.status = "failed"
            task.error = "未找到图谱"
            task.end_time = datetime.now()
            db.add(task)
            db.commit()
            return
        
        # 获取文件
        file = db.query(GraphFile).filter(GraphFile.id == file_id).first()
        if not file or not file.text_content:
            logger.error(f"未找到文件或文件无文本内容: {file_id}")
            task.status = "failed"
            task.error = "未找到文件或文件无文本内容"
            task.end_time = datetime.now()
            db.add(task)
            db.commit()
            return
        
        # 更新任务状态
        task.status = "running"
        task.start_time = datetime.now()
        db.add(task)
        db.commit()
        
        # 标记文件为抽取中状态
        file.status = "extracting"
        db.add(file)
        db.commit()
        
        # 获取Schema信息
        schema = None
        if graph.config and "schema" in graph.config:
            schema = graph.config["schema"]
            logger.info(f"获取到Schema信息: {len(schema.get('entityTypes', []))} 个实体类型, {len(schema.get('relationTypes', []))} 个关系类型")
        
        # 初始化知识抽取器
        knowledge_extractor = LLMKnowledgeExtractor(db)
        
        # 进行知识抽取
        result = await knowledge_extractor.extract_knowledge(
            graph=graph,
            text=file.text_content,
            schema=schema
        )
        
        logger.info(f"LLM知识抽取结果: {result}，success={result.get('success', False)}")
        
        # 处理结果
        if result.get("success", False):
            # 更新图谱统计信息
            update_graph_statistics(db, graph)
            
            # 更新文件状态
            file.status = "extracted"
            db.add(file)
            
            # 更新任务状态
            task.status = "completed"
            task.result = result
            task.end_time = datetime.now()
            
            # 更新任务的实体和关系计数
            if "entityCount" in result:
                task.entity_count = result["entityCount"]
            if "relationCount" in result:
                task.relation_count = result["relationCount"]
            
            db.add(task)
            db.commit()
            
            logger.info(f"LLM知识抽取成功: task_id={task.id}, 实体数={result.get('entityCount', 0)}, 关系数={result.get('relationCount', 0)}")
        else:
            # 更新任务状态为失败
            file.status = "extract_failed"
            db.add(file)
            
            task.status = "failed"
            task.error = result.get("error", "LLM知识抽取失败，未知错误")
            task.end_time = datetime.now()
            db.add(task)
            db.commit()
            
            logger.error(f"LLM知识抽取失败: task_id={task.id}, error={result.get('error')}")
    
    except Exception as e:
        logger.error(f"LLM知识抽取任务异常: {str(e)}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        
        # 尝试更新任务状态
        try:
            task = db.query(ExtractionTask).filter(ExtractionTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error = str(e)
                task.end_time = datetime.now()
                db.add(task)
                
                # 更新文件状态
                file = db.query(GraphFile).filter(GraphFile.id == file_id).first()
                if file:
                    file.status = "extract_failed"
                    db.add(file)
                
                db.commit()
        except Exception as e2:
            logger.error(f"更新任务状态失败: {str(e2)}")
    
    finally:
        db.close()

def update_graph_statistics(db: Session, graph: GraphModel) -> Dict[str, Any]:
    """
    更新图谱的Neo4j统计信息
    
    Args:
        db: 数据库会话
        graph: 图谱对象
        
    Returns:
        Dict: 统计信息
    """
    try:
        # 获取Neo4j配置
        neo4j_config = get_neo4j_config()
        if not neo4j_config or not graph.neo4j_subgraph:
            logger.warning(f"无法更新图谱统计信息: 未找到Neo4j配置或图谱无子图配置")
            return {}
            
        # 创建Neo4j服务
        neo4j_service = get_neo4j_service(
            uri=neo4j_config["uri"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            database=neo4j_config["database"],
            force_new=True
        )
        
        # 获取最新统计信息
        stats = neo4j_service.get_subgraph_statistics(graph.neo4j_subgraph)
        
        # 更新图谱
        graph.neo4j_stats = stats
        db.add(graph)
        db.commit()
        
        logger.info(f"已更新图谱统计信息: graph_id={graph.id}, 节点数={stats.get('nodeCount', 0)}, 边数={stats.get('edgeCount', 0)}")
        
        return stats
    except Exception as e:
        logger.error(f"更新图谱统计信息失败: {str(e)}")
        return {}

def get_graph_schema(db: Session, graph_id: str) -> Optional[Dict[str, Any]]:
    """
    获取知识图谱的Schema定义
    
    参数:
        db (Session): 数据库会话
        graph_id (str): 图谱ID
    
    返回:
        Optional[Dict[str, Any]]: Schema定义或None
    """
    graph = get_graph(db, graph_id)
    if not graph or not graph.config:
        return None
    
    # 从图谱配置中提取schema
    schema = graph.config.get("schema", {})
    return schema 