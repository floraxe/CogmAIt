import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.db.base import Base, get_cn_datetime
from app.utils import format_datetime


class Graph(Base):
    """
    知识图谱数据库模型
    
    存储知识图谱基本信息，如名称、描述等
    """
    __tablename__ = "graphs"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    name = Column(String(100), index=True, nullable=False)
    description = Column(String(500), nullable=True)
    entity_count = Column(Integer, default=0)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=True)
    relation_count = Column(Integer, default=0)
    config = Column(JSON, nullable=True)  # 存储图谱特定配置
    status = Column(String(20), default="active", index=True)  # active, inactive
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 添加用户ID外键
    
    # Neo4j子图相关配置
    neo4j_subgraph = Column(String(100), nullable=True)  # 子图名称
    neo4j_status = Column(String(20), default="pending", index=True)  # pending, created, error
    neo4j_config = Column(JSON, nullable=True)  # 存储Neo4j相关配置
    neo4j_stats = Column(JSON, nullable=True)  # 存储Neo4j统计信息
    dynamic_schema = Column(Boolean, default=True)  # 是否允许动态更新schema

    # 关联
    nodes = relationship("GraphNode", back_populates="graph", cascade="all, delete-orphan")
    edges = relationship("GraphEdge", back_populates="graph", cascade="all, delete-orphan")
    extraction_tasks = relationship("ExtractionTask", back_populates="graph", cascade="all, delete-orphan")
    user = relationship("User")  # 添加与User的关联
    
    def to_dict(self) -> Dict[str, Any]:
        """将知识图谱转换为字典表示形式"""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "entityCount": self.entity_count,
            "relationCount": self.relation_count,
            "config": self.config,
            "status": self.status,
            "created": format_datetime(self.created_at),
            "modified": format_datetime(self.updated_at),
            "dynamic_schema": self.dynamic_schema,
            "userId": self.user_id  # 添加用户ID到返回字典
        }
        
        # 添加模型ID（如果存在）
        if self.model_id:
            result["model_id"] = self.model_id
        
        # 添加Neo4j子图信息（如果存在）
        if self.neo4j_subgraph:
            result["neo4j_subgraph"] = self.neo4j_subgraph
            result["neo4j_status"] = self.neo4j_status
            result["neo4j_stats"] = self.neo4j_stats
            
        return result


class GraphNode(Base):
    """
    图谱节点数据库模型
    
    存储知识图谱中的节点（实体）
    """
    __tablename__ = "graph_nodes"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    graph_id = Column(String(36), ForeignKey("graphs.id"), nullable=False)
    name = Column(String(255), index=True, nullable=False)
    node_type = Column(String(50), index=True, nullable=False)  # 节点类型，如人物、组织等
    properties = Column(JSON, nullable=True)  # 节点属性
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 添加用户ID外键
    
    # 关联
    graph = relationship("Graph", back_populates="nodes")
    source_edges = relationship("GraphEdge", foreign_keys="GraphEdge.source_id", back_populates="source")
    target_edges = relationship("GraphEdge", foreign_keys="GraphEdge.target_id", back_populates="target")
    user = relationship("User")  # 添加与User的关联
    
    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为字典表示形式"""
        return {
            "id": self.id,
            "graphId": self.graph_id,
            "name": self.name,
            "type": self.node_type,
            "properties": self.properties or {},
            "description": self.description,
            "created": format_datetime(self.created_at),
            "updated": format_datetime(self.updated_at),
            "userId": self.user_id  # 添加用户ID到返回字典
        }


class GraphEdge(Base):
    """
    图谱边数据库模型
    
    存储知识图谱中的边（关系）
    """
    __tablename__ = "graph_edges"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    graph_id = Column(String(36), ForeignKey("graphs.id"), nullable=False)
    source_id = Column(String(36), ForeignKey("graph_nodes.id"), nullable=False)
    target_id = Column(String(36), ForeignKey("graph_nodes.id"), nullable=False)
    relation = Column(String(100), nullable=False)  # 关系类型
    properties = Column(JSON, nullable=True)  # 关系属性
    weight = Column(Float, default=1.0)  # 关系权重
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 添加用户ID外键
    
    # 关联
    graph = relationship("Graph", back_populates="edges")
    source = relationship("GraphNode", foreign_keys=[source_id], back_populates="source_edges")
    target = relationship("GraphNode", foreign_keys=[target_id], back_populates="target_edges")
    user = relationship("User")  # 添加与User的关联
    
    def to_dict(self) -> Dict[str, Any]:
        """将边转换为字典表示形式"""
        return {
            "id": self.id,
            "graphId": self.graph_id,
            "sourceId": self.source_id,
            "targetId": self.target_id,
            "relation": self.relation,
            "properties": self.properties or {},
            "weight": self.weight,
            "created": format_datetime(self.created_at),
            "updated": format_datetime(self.updated_at),
            "userId": self.user_id  # 添加用户ID到返回字典
        } 