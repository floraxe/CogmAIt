from fastapi import APIRouter, HTTPException, Body, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import asyncio
from ..models.agent import Agent
from ..models.user import User
from ..auth import get_current_user
from ..services import agent_service

router = APIRouter(prefix="/agent", tags=["Agent"])


class AgentCreate(BaseModel):
    name: str
    description: str
    type: str
    model_id: Optional[str] = None
    knowledge_id: Optional[str] = None
    graph_id: Optional[str] = None
    system_prompt: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    model_id: Optional[str] = None
    knowledge_id: Optional[str] = None
    graph_id: Optional[str] = None
    system_prompt: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    agent_id: str
    messages: List[Message]
    stream: bool = False


@router.get("")
async def get_agents(request: Request, current_user: User = Depends(get_current_user)):
    """获取当前用户的所有智能体列表"""
    try:
        agents = await agent_service.get_agents(current_user.id)
        return JSONResponse(
            content={
                "code": 0,
                "data": {
                    "list": agents,
                    "total": len(agents)
                },
                "message": "成功获取智能体列表"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": f"获取智能体列表失败: {str(e)}"
            }
        )


@router.get("/types")
async def get_agent_types(request: Request, current_user: User = Depends(get_current_user)):
    """获取支持的智能体类型列表"""
    try:
        types = await agent_service.get_agent_types()
        return JSONResponse(
            content={
                "code": 0,
                "data": types,
                "message": "成功获取智能体类型列表"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": f"获取智能体类型列表失败: {str(e)}"
            }
        )


@router.post("")
async def create_agent(
    request: Request,
    agent_data: AgentCreate = Body(...),
    current_user: User = Depends(get_current_user)
):
    """创建新的智能体"""
    try:
        agent = await agent_service.create_agent(
            current_user.id,
            agent_data.name,
            agent_data.description,
            agent_data.type,
            agent_data.model_id,
            agent_data.knowledge_id,
            agent_data.graph_id,
            agent_data.system_prompt,
            agent_data.config
        )
        
        return JSONResponse(
            content={
                "code": 0,
                "data": agent,
                "message": "智能体创建成功"
            }
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "code": 400,
                "message": str(e)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": f"智能体创建失败: {str(e)}"
            }
        )


@router.put("/{agent_id}")
async def update_agent(
    agent_id: str,
    agent_data: AgentUpdate = Body(...),
    current_user: User = Depends(get_current_user)
):
    """更新智能体信息"""
    try:
        updated_agent = await agent_service.update_agent(
            agent_id,
            current_user.id,
            agent_data.dict(exclude_unset=True)
        )
        
        return JSONResponse(
            content={
                "code": 0,
                "data": updated_agent,
                "message": "智能体更新成功"
            }
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "code": 400,
                "message": str(e)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": f"智能体更新失败: {str(e)}"
            }
        )


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    current_user: User = Depends(get_current_user)
):
    """删除智能体"""
    try:
        await agent_service.delete_agent(agent_id, current_user.id)
        
        return JSONResponse(
            content={
                "code": 0,
                "message": "智能体删除成功"
            }
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "code": 400,
                "message": str(e)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": f"智能体删除失败: {str(e)}"
            }
        )


@router.post("/chat")
async def chat_with_agent(
    request: Request,
    chat_request: ChatRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """与智能体对话"""
    try:
        # 转换消息格式
        messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in chat_request.messages
        ]
        
        # 处理流式响应
        if chat_request.stream:
            async def stream_response():
                async for chunk in agent_service.chat_with_agent(
                    chat_request.agent_id, 
                    current_user.id, 
                    messages,
                    stream=True
                ):
                    # 构造SSE格式的响应
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # 结束流
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering":"no"}
            )
        
        # 非流式响应
        response = await agent_service.chat_with_agent(
            chat_request.agent_id, 
            current_user.id, 
            messages
        )
        
        return JSONResponse(
            content={
                "code": 0,
                "data": response,
                "message": "对话成功"
            }
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "code": 400,
                "message": str(e)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": f"对话失败: {str(e)}"
            }
        )


@router.post("/test")
async def test_agent(
    request: Request,
    chat_request: ChatRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """测试智能体对话"""
    try:
        # 转换消息格式
        messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in chat_request.messages
        ]
        
        # 处理流式响应
        if chat_request.stream:
            async def stream_response():
                async for chunk in agent_service.test_agent(
                    chat_request.agent_id, 
                    current_user.id, 
                    messages,
                    stream=True
                ):
                    # 构造SSE格式的响应
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # 结束流
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering":"no"}
            )
        
        # 非流式响应
        response = await agent_service.test_agent(
            chat_request.agent_id, 
            current_user.id, 
            messages
        )
        
        return JSONResponse(
            content={
                "code": 0,
                "data": response,
                "message": "测试对话成功"
            }
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "code": 400,
                "message": str(e)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": f"测试对话失败: {str(e)}"
            }
        ) 