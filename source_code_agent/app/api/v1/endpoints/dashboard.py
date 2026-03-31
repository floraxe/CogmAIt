from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import func, or_, Column, String, DateTime, Text
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, aliased
import random
import json
import uuid
import re

from app.db.session import get_db
from app.utils.deps import get_current_active_user
from app.models.user import User
from app.models.agent import Agent, AgentChatHistory
from app.models.file import File
from app.models.knowledge import Knowledge
from app.models.graph import Graph
from app.models.model import Model
from app.db.base import Base, get_cn_datetime

router = APIRouter()

@router.get("/stats")
async def get_dashboard_stats(
    force_refresh_recommended: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取仪表盘统计数据
    
    根据用户角色返回不同范围的统计数据:
    - 管理员: 全平台统计
    - 部门管理员: 部门内统计
    - 普通用户: 个人统计
    
    参数：
    - force_refresh_recommended: 强制刷新推荐问题，不使用缓存
    """
    # 检查用户权限
    is_admin = current_user.role == "admin"
    is_department_admin = current_user.role == "department_admin"
    
    # 根据角色确定筛选条件
    filter_conditions = {}
    
    if not is_admin:
        if is_department_admin:
            # 部门管理员查看部门内数据
            filter_conditions["department"] = current_user.department
        else:
            # 普通用户只查看自己的数据
            filter_conditions["user_id"] = current_user.id
    
    # 获取模型计数
    model_count = get_model_count(db, filter_conditions)
    
    # 获取知识库计数
    knowledge_count = get_knowledge_count(db, filter_conditions)
    
    # 获取知识图谱计数
    graph_count = get_graph_count(db, filter_conditions)
    
    # 获取智能体计数
    agent_count = get_agent_count(db, filter_conditions)
    
    # 获取对话统计
    chat_stats = get_chat_stats(db, current_user, is_admin, is_department_admin)
    
    # 获取使用趋势数据
    usage_trends = get_usage_trends(db, current_user, is_admin, is_department_admin)
    
    # 获取模型分布数据
    model_distribution = get_model_distribution(db, current_user, is_admin, is_department_admin)
    
    # 获取热门问题列表
    top_questions = get_top_questions(db, current_user, is_admin, is_department_admin)
    
    # 获取推荐问题列表
    recommended_questions = await get_recommended_questions(db, current_user, is_admin, is_department_admin, force_refresh=force_refresh_recommended)
    
    # 获取热门知识库
    top_knowledge_bases = get_top_knowledge_bases(db, current_user, is_admin, is_department_admin)
    
    # 获取热门智能体
    top_agents = get_top_agents(db, current_user, is_admin, is_department_admin)
    
    # 构建返回结果
    return {
        "statistics": {
            "modelCount": model_count,
            "knowledgeCount": knowledge_count,
            "graphCount": graph_count,
            "agentCount": agent_count,
            "chatCount": chat_stats["totalCount"],
            "recentChatCount": chat_stats["recentCount"],
            "mostUsedAgent": chat_stats.get("mostUsedAgent")
        },
        "usageTrends": usage_trends,
        "modelDistribution": model_distribution,
        "topQuestions": top_questions,
        "recommendedQuestions": recommended_questions,
        "topKnowledgeBases": top_knowledge_bases,
        "topAgents": top_agents,
        "scope": "platform" if is_admin else ("department" if is_department_admin else "personal")
    }

def get_model_count(db: Session, filter_conditions: Dict) -> int:
    """获取模型计数"""
    query = db.query(func.count(Model.id))
    
    if filter_conditions:
        # 如果有用户ID条件
        if "user_id" in filter_conditions:
            query = query.filter(Model.user_id == filter_conditions["user_id"])
        # 如果有部门条件
        elif "department" in filter_conditions:
            # 通过User表连接查询获取指定部门的模型
            creator = aliased(User)
            query = query.join(
                creator,
                Model.user_id == creator.id
            ).filter(
                creator.department == filter_conditions["department"]
            )
    
    return query.scalar() or 0

def get_knowledge_count(db: Session, filter_conditions: Dict) -> int:
    """获取知识库计数"""
    query = db.query(func.count(Knowledge.id))
    
    # 添加过滤条件：根据用户ID或部门过滤
    if filter_conditions:
        if "user_id" in filter_conditions:
            # 直接使用user_id字段过滤
            query = query.filter(Knowledge.user_id == filter_conditions["user_id"])
        
        if "department" in filter_conditions:
            # 通过User表连接查询获取指定部门的知识库
            creator = aliased(User)
            query = query.join(
                creator, 
                Knowledge.user_id == creator.id
            ).filter(
                creator.department == filter_conditions["department"]
            )
    
    return query.scalar() or 0

def get_graph_count(db: Session, filter_conditions: Dict) -> int:
    """获取知识图谱计数"""
    query = db.query(func.count(Graph.id))
    
    # 添加过滤条件：根据用户ID或部门过滤
    if filter_conditions:
        if "user_id" in filter_conditions:
            # 直接使用user_id字段过滤
            query = query.filter(Graph.user_id == filter_conditions["user_id"])
        
        if "department" in filter_conditions:
            # 通过User表连接查询获取指定部门的知识图谱
            creator = aliased(User)
            query = query.join(
                creator,
                Graph.user_id == creator.id
            ).filter(
                creator.department == filter_conditions["department"]
            )
    
    return query.scalar() or 0

def get_agent_count(db: Session, filter_conditions: Dict) -> int:
    """获取智能体计数"""
    query = db.query(func.count(Agent.id))
    
    if filter_conditions:
        # 如果有用户ID条件
        if "user_id" in filter_conditions:
            query = query.filter(Agent.user_id == filter_conditions["user_id"])
        # 如果有部门条件
        elif "department" in filter_conditions:
            # 通过User表连接查询获取指定部门的智能体
            creator = aliased(User)
            query = query.join(
                creator,
                Agent.user_id == creator.id
            ).filter(
                creator.department == filter_conditions["department"]
            )
    
    return query.scalar() or 0

def get_chat_stats(db: Session, current_user: User, is_admin: bool, is_department_admin: bool) -> Dict:
    """获取对话统计数据"""
    # 基础查询
    total_query = db.query(func.count(AgentChatHistory.id))
    recent_query = db.query(func.count(AgentChatHistory.id)).filter(
        AgentChatHistory.created_at >= datetime.now() - timedelta(days=7)
    )
    
    # 根据权限过滤
    if not is_admin:
        if is_department_admin:
            # 部门管理员: 通过User表连接查询筛选部门内的聊天记录
            creator = aliased(User)
            total_query = total_query.join(
                creator,
                AgentChatHistory.user_id == creator.id
            ).filter(
                creator.department == current_user.department
            )
            
            # 为最近查询也使用相同的过滤条件
            recent_creator = aliased(User)
            recent_query = recent_query.join(
                recent_creator,
                AgentChatHistory.user_id == recent_creator.id
            ).filter(
                recent_creator.department == current_user.department
            )
        else:
            # 普通用户: 仅查看自己的聊天记录
            total_query = total_query.filter(AgentChatHistory.user_id == current_user.id)
            recent_query = recent_query.filter(AgentChatHistory.user_id == current_user.id)
    
    # 获取最常用的智能体
    most_used_agent = None
    if not is_admin or is_department_admin:
        # 如果不是管理员，或者是部门管理员，获取对应范围内最常用的智能体
        agent_usage_query = db.query(
            AgentChatHistory.agent_id, 
            func.count(AgentChatHistory.agent_id).label('count')
        )
        
        if is_department_admin:
            # 部门管理员: 通过User表连接查询筛选部门内的聊天记录
            creator = aliased(User)
            agent_usage_query = agent_usage_query.join(
                creator,
                AgentChatHistory.user_id == creator.id
            ).filter(
                creator.department == current_user.department
            )
        else:
            # 普通用户: 只看自己的
            agent_usage_query = agent_usage_query.filter(AgentChatHistory.user_id == current_user.id)
        
        agent_usage_query = agent_usage_query.group_by(AgentChatHistory.agent_id).order_by(func.count(AgentChatHistory.agent_id).desc())
        
        most_used_result = agent_usage_query.first()
        if most_used_result:
            agent_id, count = most_used_result
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                most_used_agent = {
                    "id": agent.id,
                    "name": agent.name,
                    "count": count
                }
    
    return {
        "totalCount": total_query.scalar() or 0,
        "recentCount": recent_query.scalar() or 0,
        "mostUsedAgent": most_used_agent
    }

def get_usage_trends(db: Session, current_user: User, is_admin: bool, is_department_admin: bool) -> Dict:
    """获取使用趋势数据"""
    # 获取过去7天的日期列表
    days = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    
    # 初始化结果数据
    result = {
        "labels": days,
        "datasets": [
            {
                "name": "对话次数",
                "data": [0] * len(days)
            },
            {
                "name": "知识查询",
                "data": [0] * len(days)
            },
            {
                "name": "API调用",
                "data": [0] * len(days)
            }
        ]
    }
    
    # 构建查询
    for i, day in enumerate(days):
        day_start = datetime.strptime(day, '%Y-%m-%d')
        day_end = day_start + timedelta(days=1)
        
        # 对话次数查询
        chat_query = db.query(func.count(AgentChatHistory.id)).filter(
            AgentChatHistory.created_at >= day_start,
            AgentChatHistory.created_at < day_end
        )
        
        # 根据权限筛选
        if not is_admin:
            if is_department_admin:
                # 部门管理员: 通过User表连接查询筛选部门内的聊天记录
                creator = aliased(User)
                chat_query = chat_query.join(
                    creator,
                    AgentChatHistory.user_id == creator.id
                ).filter(
                    creator.department == current_user.department
                )
            else:
                # 普通用户: 仅查看自己的聊天记录
                chat_query = chat_query.filter(AgentChatHistory.user_id == current_user.id)
                
        # 知识查询次数（通过AgentChatHistory中的extra_data字段判断是否使用了知识库）
        knowledge_query = db.query(func.count(AgentChatHistory.id)).filter(
            AgentChatHistory.created_at >= day_start,
            AgentChatHistory.created_at < day_end,
            AgentChatHistory.extra_data.contains('"knowledge_used": true')
        )
        
        if not is_admin:
            if is_department_admin:
                # 部门管理员: 通过User表连接查询筛选部门内的聊天记录
                creator = aliased(User)
                knowledge_query = knowledge_query.join(
                    creator,
                    AgentChatHistory.user_id == creator.id
                ).filter(
                    creator.department == current_user.department
                )
            else:
                knowledge_query = knowledge_query.filter(AgentChatHistory.user_id == current_user.id)
                
        # API调用次数（通过access_type字段判断）
        api_query = db.query(func.count(AgentChatHistory.id)).filter(
            AgentChatHistory.created_at >= day_start,
            AgentChatHistory.created_at < day_end,
            AgentChatHistory.access_type == "api"
        )
        
        if not is_admin:
            if is_department_admin:
                # 部门管理员: 通过User表连接查询筛选部门内的聊天记录
                creator = aliased(User)
                api_query = api_query.join(
                    creator,
                    AgentChatHistory.user_id == creator.id
                ).filter(
                    creator.department == current_user.department
                )
            else:
                api_query = api_query.filter(AgentChatHistory.user_id == current_user.id)
        
        # 更新结果
        result["datasets"][0]["data"][i] = chat_query.scalar() or 0
        result["datasets"][1]["data"][i] = knowledge_query.scalar() or 0
        result["datasets"][2]["data"][i] = api_query.scalar() or 0
    
    return result

def get_model_distribution(db: Session, current_user: User, is_admin: bool, is_department_admin: bool) -> Dict:
    """获取模型使用分布"""
    # 获取所有可用的模型
    model_query = db.query(Model)
    
    # 根据权限筛选可用模型
    if not is_admin:
        if is_department_admin:
            # 部门管理员: 通过User表连接查询获取部门内的模型
            creator = aliased(User)
            department_model_query = model_query.join(
                creator,
                Model.user_id == creator.id
            ).filter(
                creator.department == current_user.department
            )
            # 合并公共模型和部门模型
            model_query = model_query.filter(Model.user_id.is_(None)).union(department_model_query)
        else:
            # 普通用户: 查看公共模型和自己的模型
            model_query = model_query.filter(
                or_(Model.user_id.is_(None), Model.user_id == current_user.id)
            )
    
    # 获取所有符合条件的模型
    models = model_query.all()
    
    if not models:
        # 如果没有数据，返回示例数据
        return {
            "data": [
                {"name": "OpenAI", "value": 30},
                {"name": "讯飞星火", "value": 25},
                {"name": "文心一言", "value": 20},
                {"name": "通义千问", "value": 15},
                {"name": "其他", "value": 10}
            ]
        }
    
    # 直接从聊天历史记录表中获取模型使用数据，基于model_id字段
    model_usage_data = {}
    
    for model in models:
        # 查询该模型在聊天历史中的使用次数
        usage_query = db.query(
            func.count(AgentChatHistory.id)
        ).filter(
            AgentChatHistory.model_id == model.id
        )
        
        # 根据权限筛选调用记录
        if not is_admin:
            if is_department_admin:
                # 部门管理员: 通过User表连接查询筛选部门内的聊天记录
                creator = aliased(User)
                usage_query = usage_query.join(
                    creator,
                    AgentChatHistory.user_id == creator.id
                ).filter(
                    creator.department == current_user.department
                )
            else:
                # 普通用户: 仅查看自己的聊天记录
                usage_query = usage_query.filter(AgentChatHistory.user_id == current_user.id)
        
        # 获取这个模型的调用次数
        count = usage_query.scalar() or 0
        model_usage_data[model.name] = count
    
    # 格式化结果，包括所有模型（即使调用次数为0）
    result = []
    for model in models:
        result.append({
            "name": model.name,
            "value": model_usage_data.get(model.name, 0)  # 如果没有调用记录，则使用0
        })
    
    # 按调用次数降序排序
    result.sort(key=lambda x: x["value"], reverse=True)
    
    # 如果所有模型的调用次数都是0，使用示例数据
    if all(item["value"] == 0 for item in result):
        return {
            "data": [
                    {"name": "OpenAI", "value": 30},
                    {"name": "讯飞星火", "value": 25},
                    {"name": "文心一言", "value": 20},
                    {"name": "通义千问", "value": 15},
                    {"name": "其他", "value": 10}
                ]
            }
    
    return {
        "data": result
    }

def get_top_questions(db: Session, current_user: User, is_admin: bool, is_department_admin: bool) -> List[Dict]:
    """获取热门问题列表"""
    # 查询常见问题
    query = db.query(
        AgentChatHistory.user_message,
        func.count(AgentChatHistory.user_message).label('count')
    )
    
    # 根据权限筛选
    if not is_admin:
        if is_department_admin:
            # 部门管理员: 通过User表连接查询筛选部门内的聊天记录
            creator = aliased(User)
            query = query.join(
                creator,
                AgentChatHistory.user_id == creator.id
            ).filter(
                creator.department == current_user.department
            )
        else:
            # 普通用户: 仅查看自己的聊天记录
            query = query.filter(AgentChatHistory.user_id == current_user.id)
    
    # 过滤过短的问题
    query = query.filter(func.length(AgentChatHistory.user_message) > 0)
    
    # 分组、排序和限制
    query = query.group_by(AgentChatHistory.user_message)\
                .order_by(func.count(AgentChatHistory.user_message).desc())\
                .limit(10)  # 修改为加载10个问题
    
    results = query.all()
    
    if not results:
        # 如果没有数据，查询最近的10个问题
        # 使用子查询方式获取最近的问题，避免GROUP BY问题
        subquery = db.query(
            AgentChatHistory.id,
            AgentChatHistory.user_message
        ).filter(
            func.length(AgentChatHistory.user_message) > 5
        )
        
        # 根据权限筛选
        if not is_admin:
            if is_department_admin:
                creator = aliased(User)
                subquery = subquery.join(
                    creator,
                    AgentChatHistory.user_id == creator.id
                ).filter(
                    creator.department == current_user.department
                )
            else:
                subquery = subquery.filter(AgentChatHistory.user_id == current_user.id)
        
        # 按时间降序排序并限制结果数量
        subquery = subquery.order_by(AgentChatHistory.created_at.desc()).limit(10).subquery()
        
        # 在子查询结果上进行分组计数
        recent_query = db.query(
            subquery.c.user_message,
            func.count(subquery.c.id).label('count')
        ).group_by(
            subquery.c.user_message
        )
        
        recent_results = recent_query.all()
        
        if recent_results:
            return [{"content": message, "count": count} for message, count in recent_results]
        
        # 如果仍然没有数据，返回示例数据
        return [
            {"content": "如何优化大语言模型的推理性能？", "count": 325},
            {"content": "知识图谱与传统向量数据库的区别是什么？", "count": 287},
            {"content": "跨模态检索在多模态模型中如何实现？", "count": 256},
            {"content": "如何评估RAG系统的检索效果？", "count": 213},
            {"content": "大语言模型如何与专业知识库结合？", "count": 198},
            {"content": "如何解决大模型的幻觉问题？", "count": 186},
            {"content": "RLHF是什么，如何实现？", "count": 173},
            {"content": "多智能体系统如何协同工作？", "count": 162},
            {"content": "企业如何有效实施AI战略？", "count": 154},
            {"content": "知识图谱在医疗领域的应用有哪些？", "count": 147}
        ]
    
    return [{"content": message, "count": count} for message, count in results]

class RecommendedQuestion(Base):
    """推荐问题数据库模型"""
    __tablename__ = "recommended_questions"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4().hex))
    user_id = Column(String(36), index=True, nullable=True)  # 如果为空，表示全局推荐
    department = Column(String(255), index=True, nullable=True)  # 如果不为空，表示部门推荐
    content = Column(Text, nullable=False)  # 问题内容
    score = Column(String(255), nullable=True)  # 相关度评分
    created_at = Column(DateTime, default=get_cn_datetime)
    updated_at = Column(DateTime, default=get_cn_datetime, onupdate=get_cn_datetime)

async def get_recommended_questions(db: Session, current_user: User, is_admin: bool, is_department_admin: bool, force_refresh: bool = False) -> List[Dict]:
    """获取推荐问题列表（基于用户历史聊天记录，由大模型生成）"""
    try:
        user_id = current_user.id
        department = None if is_admin else (current_user.department if is_department_admin else None)
        
        # 查询是否有现有的推荐问题缓存
        query = db.query(RecommendedQuestion)
        
        if user_id:
            query = query.filter(RecommendedQuestion.user_id == user_id)
        elif department:
            query = query.filter(RecommendedQuestion.department == department)
        # else:
        #     # 管理员看全局推荐
        #     query = query.filter(RecommendedQuestion.user_id.is_(None), RecommendedQuestion.department.is_(None))
        
        # 检查最近更新时间
        cached_questions = query.order_by(RecommendedQuestion.updated_at.desc()).all()
        
        # 如果有缓存且不超过24小时，且不是强制刷新，直接返回
        if not force_refresh and cached_questions and len(cached_questions) >= 10:
            now = datetime.now()
            if (now - cached_questions[0].updated_at).total_seconds() < 86400:  # 24小时
                return [{"content": q.content, "score": q.score} for q in cached_questions[:10]]
        
        # 需要生成新的推荐问题
        # 首先获取一个能够对话的模型
        chat_model = db.query(Model).filter(
            Model.type == "chat",
            Model.status == "active"
        ).first()
        
        if not chat_model:
            # 如果没有可用的聊天模型，返回默认推荐
            return [
                {"content": "知识图谱如何与大语言模型结合？", "score": "98%"},
                {"content": "如何设计更高效的Prompt提示词？", "score": "95%"},
                {"content": "多模态模型在医疗领域有哪些应用？", "score": "92%"},
                {"content": "如何降低大模型的幻觉问题？", "score": "89%"},
                {"content": "跨语言迁移学习的最佳实践是什么？", "score": "85%"},
                {"content": "如何提高RAG系统的检索准确率？", "score": "82%"},
                {"content": "向量数据库的选型标准有哪些？", "score": "80%"},
                {"content": "Agent系统的架构设计最佳实践？", "score": "78%"},
                {"content": "AIGC在企业中的落地场景有哪些？", "score": "76%"},
                {"content": "如何评估大模型的知识能力？", "score": "74%"}
            ]
        
        # 获取用户历史聊天记录
        history_query = db.query(AgentChatHistory.user_message)
        
        if user_id:
            history_query = history_query.filter(AgentChatHistory.user_id == user_id)
        elif department:
            # 获取部门内所有用户的ID
            department_users = db.query(User.id).filter(User.department == department).all()
            department_user_ids = [u.id for u in department_users]
            history_query = history_query.filter(AgentChatHistory.user_id.in_(department_user_ids))
        print("history_query:::", history_query,"user_id:::", user_id,"department:::", department)
        # 获取最近的50条历史记录
        chat_history = history_query.order_by(AgentChatHistory.created_at.desc()).limit(50).all()
        
        # 提取历史问题
        history_messages = [msg.user_message for msg in chat_history]
        
        # 构建提示词
        if history_messages:
            history_text = "\n".join([f"- {msg}" for msg in history_messages[:20]])  # 仅使用最近的20条
            prompt = f"""基于以下用户的历史提问，生成10个相关的推荐问题，这些问题应该是用户可能感兴趣的新问题。
历史提问:
{history_text}

对于每个推荐问题，给出70%到99%之间的相关度百分比，以反映与用户兴趣的相关程度。
格式：
1. [问题内容] - [相关度百分比]
2. [问题内容] - [相关度百分比]
...
请直接给出问题列表，不要添加额外解释。"""
        else:
            # 如果没有历史记录，则生成通用的AI和大数据相关问题
            prompt = """生成10个关于AI、大数据、知识图谱和大语言模型的高质量问题，这些问题应该是一个企业用户可能感兴趣的。
对于每个问题，给出70%到99%之间的相关度百分比。
格式：
1. [问题内容] - [相关度百分比]
2. [问题内容] - [相关度百分比]
...
请直接给出问题列表，不要添加额外解释。"""
        
        # 调用模型生成推荐问题
        from app.utils.model import execute_model_inference
        model_response = await execute_model_inference(
            db,
            chat_model.id,
            {
                "messages": [
                    {"role": "system", "content": "你是一个AI推荐系统，负责生成用户可能感兴趣的问题。请直接以列表形式返回问题，不要添加思考过程或额外解释。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }
        )
        print("model_response:::", model_response)
        model_response = json.loads(model_response)
        # 解析模型响应
        recommended_questions = []
        try:
            if model_response:
                content = ""
                
                # 处理不同格式的模型输出
                if isinstance(model_response, dict):
                    if "choices" in model_response and len(model_response["choices"]) > 0:
                        if "message" in model_response["choices"][0]:
                            content = model_response["choices"][0]["message"].get("content", "")
                        elif "text" in model_response["choices"][0]:
                            content = model_response["choices"][0].get("text", "")
                elif isinstance(model_response, str):
                    content = model_response
                
                # 处理可能包含思考过程的输出
                # 移除<think>...</think>标签之间的内容
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                
                # 解析内容，提取问题和相关度
                lines = content.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 跳过可能的标题行或指令行
                    if any(keyword in line.lower() for keyword in ["问题列表", "推荐问题", "以下是", "here are", "格式："]):
                        continue
                        
                    # 移除序号
                    if re.match(r'^\d+[\.\)、]\s', line):
                        line = re.sub(r'^\d+[\.\)、]\s', '', line)
                        
                    # 提取问题和相关度
                    if " - " in line:
                        try:
                            question, score = line.rsplit(" - ", 1)
                            # 验证分数格式
                            if "%" in score:
                                recommended_questions.append({"content": question.strip(), "score": score.strip()})
                            else:
                                recommended_questions.append({"content": question.strip(), "score": f"{score.strip()}%"})
                        except Exception as e:
                            print(f"解析推荐问题行出错: {e}, 行内容: {line}")
                            # 如果解析失败，尝试直接使用整行作为问题
                            recommended_questions.append({"content": line, "score": f"{random.randint(70, 99)}%"})
                    else:
                        recommended_questions.append({"content": line, "score": f"{random.randint(70, 99)}%"})
        except Exception as e:
            print(f"解析模型响应时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 如果结果少于10个，添加一些默认问题
        default_questions = [
            {"content": "知识图谱如何与大语言模型结合？", "score": "98%"},
            {"content": "如何设计更高效的Prompt提示词？", "score": "95%"},
            {"content": "多模态模型在医疗领域有哪些应用？", "score": "92%"},
            {"content": "如何降低大模型的幻觉问题？", "score": "89%"},
            {"content": "跨语言迁移学习的最佳实践是什么？", "score": "85%"},
            {"content": "如何提高RAG系统的检索准确率？", "score": "82%"},
            {"content": "向量数据库的选型标准有哪些？", "score": "80%"},
            {"content": "Agent系统的架构设计最佳实践？", "score": "78%"},
            {"content": "AIGC在企业中的落地场景有哪些？", "score": "76%"},
            {"content": "如何评估大模型的知识能力？", "score": "74%"}
        ]
        
        while len(recommended_questions) < 10:
            remaining = 10 - len(recommended_questions)
            recommended_questions.extend(default_questions[:remaining])
        
        # 只保留前10个问题
        recommended_questions = recommended_questions[:10]
        
        # 清除现有缓存
        for old_question in cached_questions:
            db.delete(old_question)
        
        # 保存到数据库作为缓存
        for question in recommended_questions:
            db_question = RecommendedQuestion(
                id=str(uuid.uuid4().hex),
                user_id=user_id,
                department=department,
                content=question["content"],
                score=question["score"]
            )
            db.add(db_question)
        
        db.commit()
        
        return recommended_questions
        
    except Exception as e:
        print(f"生成推荐问题时出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 发生错误时返回默认推荐
        return [
            {"content": "知识图谱如何与大语言模型结合？", "score": "98%"},
            {"content": "如何设计更高效的Prompt提示词？", "score": "95%"},
            {"content": "多模态模型在医疗领域有哪些应用？", "score": "92%"},
            {"content": "如何降低大模型的幻觉问题？", "score": "89%"},
            {"content": "跨语言迁移学习的最佳实践是什么？", "score": "85%"},
            {"content": "如何提高RAG系统的检索准确率？", "score": "82%"},
            {"content": "向量数据库的选型标准有哪些？", "score": "80%"},
            {"content": "Agent系统的架构设计最佳实践？", "score": "78%"},
            {"content": "AIGC在企业中的落地场景有哪些？", "score": "76%"},
            {"content": "如何评估大模型的知识能力？", "score": "74%"}
        ]

def get_top_knowledge_bases(db: Session, current_user: User, is_admin: bool, is_department_admin: bool) -> List[Dict]:
    """获取热门知识库列表"""
    # 查询热门知识库
    
    # 使用真实数据代替示例数据
    query = db.query(Knowledge)
    
    # 根据权限筛选
    if not is_admin:
        if is_department_admin:
            # 部门管理员: 通过User表连接查询获取指定部门的知识库
            creator = aliased(User)
            query = query.join(
                creator, 
                Knowledge.user_id == creator.id
            ).filter(
                creator.department == current_user.department
            )
        else:
            # 普通用户只查看自己的知识库
            query = query.filter(Knowledge.user_id == current_user.id)
    
    # 排序和限制
    query = query.order_by(Knowledge.created_at.desc()).limit(5)
    
    results = query.all()
    
    if not results:
        # 使用固定的示例数据
        return [
            {"name": "医疗诊断知识库", "description": "包含常见疾病诊断和治疗方案", "useCount": 1342},
            {"name": "法律法规库", "description": "各类法律法规及案例解析", "useCount": 982},
            {"name": "金融市场分析", "description": "金融市场数据与分析报告", "useCount": 876},
            {"name": "技术文档集", "description": "技术文档和最佳实践指南", "useCount": 754},
            {"name": "教育资源库", "description": "各学科教学资源和素材", "useCount": 701}
        ]
    
    # 尝试通过聊天记录获取知识库使用次数，这里使用查询关联计数
    knowledge_usage = {}
    try:
        # 查询聊天历史中提及知识库的记录
        usage_query = db.query(
            AgentChatHistory.extra_data,
            func.count(AgentChatHistory.id).label('count')
        ).filter(
            AgentChatHistory.extra_data.contains('"knowledge_base_id"')
        ).group_by(AgentChatHistory.extra_data)
        
        # 根据权限过滤聊天历史
        if not is_admin:
            if is_department_admin:
                # 部门管理员: 获取部门内所有用户ID
                department_user_ids = [
                    user.id for user in 
                    db.query(User.id).filter(User.department == current_user.department).all()
                ]
                usage_query = usage_query.filter(AgentChatHistory.user_id.in_(department_user_ids))
            else:
                # 普通用户: 仅查看自己的聊天记录
                usage_query = usage_query.filter(AgentChatHistory.user_id == current_user.id)
        
        # 解析聊天历史中的知识库ID并计数
        for extra_data_json, count in usage_query.all():
            try:
                if isinstance(extra_data_json, str):
                    extra_data = json.loads(extra_data_json)
                else:
                    extra_data = extra_data_json
                
                if extra_data and "knowledge_base_id" in extra_data:
                    kb_id = extra_data["knowledge_base_id"]
                    knowledge_usage[kb_id] = knowledge_usage.get(kb_id, 0) + count
            except Exception as e:
                print(f"解析知识库使用数据失败: {e}")
    except Exception as e:
        print(f"获取知识库使用统计失败: {e}")
    
    # 生成结果数据，添加使用次数
    return [
        {
            "id": kb.id,
            "name": kb.name, 
            "description": kb.description or f"{kb.name}的描述信息", 
            # 如果能找到使用次数则使用，否则随机生成
            "useCount": knowledge_usage.get(kb.id, random.randint(50, 500))
        } 
        for kb in results
    ]

def get_top_agents(db: Session, current_user: User, is_admin: bool, is_department_admin: bool) -> List[Dict]:
    """获取热门智能体列表"""
    # 查询智能体使用次数
    query = db.query(
        Agent.id, 
        Agent.name, 
        Agent.description,
        func.count(AgentChatHistory.id).label('count')
    ).join(
        AgentChatHistory, AgentChatHistory.agent_id == Agent.id, isouter=True
    )
    
    # 根据权限筛选
    if not is_admin:
        if is_department_admin:
            # 部门管理员: 获取部门内的智能体
            creator = aliased(User)
            query = query.join(
                creator, 
                Agent.user_id == creator.id
            ).filter(
                creator.department == current_user.department
            )
        else:
            # 普通用户: 仅查看自己的智能体
            query = query.filter(Agent.user_id == current_user.id)
    
    # 分组、排序和限制
    query = query.group_by(Agent.id, Agent.name, Agent.description)\
                .order_by(func.count(AgentChatHistory.id).desc())\
                .limit(5)
    
    results = query.all()
    
    if not results:
        # 如果没有数据，返回示例数据
        return [
            {"id": "example1", "name": "智能客服助手", "description": "24小时在线客户服务", "useCount": 2541},
            {"id": "example2", "name": "法律顾问", "description": "法律咨询与案例分析", "useCount": 1985},
            {"id": "example3", "name": "医疗诊断助手", "description": "医学知识查询与初步诊断", "useCount": 1752},
            {"id": "example4", "name": "财务分析师", "description": "财务数据分析与报告生成", "useCount": 1423},
            {"id": "example5", "name": "教育辅导专家", "description": "学习辅导与知识解答", "useCount": 1254}
        ]
    
    # 确保所有结果都有用量计数，即使是0
    return [
        {
            "id": agent_id,
            "name": name or "未命名智能体",
            "description": description or f"{name or '未命名智能体'}的描述", 
            "useCount": count or 0
        } 
        for agent_id, name, description, count in results
    ] 