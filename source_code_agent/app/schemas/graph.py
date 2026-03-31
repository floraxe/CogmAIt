from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union


class GraphBase(BaseModel):
    """知识图谱基础模式"""
    name: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class GraphCreate(GraphBase):
    """创建知识图谱请求模式"""
    pass


class GraphUpdate(BaseModel):
    """更新知识图谱请求模式"""
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


class GraphResponse(GraphBase):
    """知识图谱响应模式"""
    id: str
    entity_count: int
    relation_count: int
    status: str
    created: str = Field(alias="created_at")
    modified: Optional[str] = Field(default=None, alias="updated_at")
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
        alias_generator=lambda field_name: ''.join(
            x if i == 0 else x.capitalize() for i, x in enumerate(field_name.split('_'))
        )
    )


class GraphListResponse(BaseModel):
    """知识图谱列表响应"""
    total: int
    items: List[GraphResponse]


class GraphNodeBase(BaseModel):
    """图谱节点基础模式"""
    name: str
    node_type: str
    properties: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class GraphNodeCreate(GraphNodeBase):
    """创建图谱节点请求模式"""
    pass


class GraphNodeUpdate(BaseModel):
    """更新图谱节点请求模式"""
    name: Optional[str] = None
    node_type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class GraphNodeResponse(GraphNodeBase):
    """图谱节点响应模式"""
    id: str
    graph_id: str
    created: str = Field(alias="created_at")
    updated: Optional[str] = Field(default=None, alias="updated_at")
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
        alias_generator=lambda field_name: ''.join(
            x if i == 0 else x.capitalize() for i, x in enumerate(field_name.split('_'))
        )
    )


class GraphNodeListResponse(BaseModel):
    """图谱节点列表响应"""
    total: int
    items: List[GraphNodeResponse]


class GraphEdgeBase(BaseModel):
    """图谱边基础模式"""
    source_id: str
    target_id: str
    relation_type: str
    properties: Optional[Dict[str, Any]] = None
    weight: float = 1.0
    bidirectional: bool = False


class GraphEdgeCreate(GraphEdgeBase):
    """创建图谱边请求模式"""
    pass


class GraphEdgeUpdate(BaseModel):
    """更新图谱边请求模式"""
    relation_type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    weight: Optional[float] = None
    bidirectional: Optional[bool] = None


class GraphEdgeResponse(GraphEdgeBase):
    """图谱边响应模式"""
    id: str
    graph_id: str
    created: str = Field(alias="created_at")
    updated: Optional[str] = Field(default=None, alias="updated_at")
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
        alias_generator=lambda field_name: ''.join(
            x if i == 0 else x.capitalize() for i, x in enumerate(field_name.split('_'))
        )
    )


class GraphEdgeListResponse(BaseModel):
    """图谱边列表响应"""
    total: int
    items: List[GraphEdgeResponse]


class GraphVisualizationNode(BaseModel):
    """可视化节点模式"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphVisualizationEdge(BaseModel):
    """可视化边模式"""
    id: str
    source: str
    target: str
    type: str
    weight: float = 1.0
    bidirectional: bool = False
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphVisualizationResponse(BaseModel):
    """图谱可视化数据响应"""
    nodes: List[GraphVisualizationNode]
    edges: List[GraphVisualizationEdge] 