from typing import Dict, List, Any, Optional

from app.utils.model import execute_model_inference
from app.models.model import Model
import json

class AISQLAssistant:
    """AI SQL助手，使用大模型生成SQL查询语句"""
    
    def __init__(self, model: Model):
        """
        初始化SQL助手
        
        参数:
            model: 要使用的AI模型
        """
        self.model = model
    
    async def generate_sql(self, db_type: str, db_structure: Dict[str, Any], user_query: str, db) -> str:
        """
        根据用户自然语言查询和数据库结构生成SQL查询语句
        
        参数:
            db_type: 数据库类型，如mysql、postgresql等
            db_structure: 数据库结构信息
            user_query: 用户的自然语言查询
            db: 数据库会话
            
        返回:
            str: 生成的SQL查询语句
        """
        # 检查用户查询是否为None或空
        if not user_query:
            raise ValueError("用户查询不能为空")
        
        # 构建适合大模型的数据库结构描述
        db_description = self._build_db_description(db_structure)
        
        # 构建提示语
        prompt = self._build_prompt(db_type, db_description, user_query)
        
        # 使用大模型生成SQL
        model_response = await execute_model_inference(
            db,  # 传入数据库会话
            self.model.id,
            {
                "messages": [
                    {"role": "system", "content": "你是一个专业的SQL生成助手。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # 使用较低的温度以获得更精确的SQL
                "max_tokens": 1500,
                "model_type": "chat"
            }
        )
        model_response = json.loads(model_response)
        print("SQL model_response:::", model_response,type(model_response))
        # 检查响应是否有错误
        # if "error" in model_response:
        #     raise Exception(f"模型调用失败: {model_response['error']}")
        # 提取生成的SQL查询
        print("model_response:::", model_response,type(model_response))
        response_content = model_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        sql_query = self._extract_sql_from_response(response_content)
        
        return sql_query
    
    def _build_db_description(self, db_structure: Dict[str, Any]) -> str:
        """根据数据库结构构建描述文本"""
        description = f"数据库名: {db_structure.get('database', 'unknown')}\n\n"
        description += "表结构:\n"
        
        for table_name, table_info in db_structure.get("tables", {}).items():
            description += f"\n表名: {table_name}\n"
            description += "列:\n"
            
            for column in table_info.get("columns", []):
                nullable = "可为空" if column.get("nullable") else "非空"
                default = f", 默认值: {column.get('default')}" if column.get("default") else ""
                description += f"- {column.get('name')} ({column.get('type')}, {nullable}{default})\n"
            
            if table_info.get("foreign_keys"):
                description += "\n外键关系:\n"
                for fk in table_info.get("foreign_keys", []):
                    description += f"- {fk.get('column')} -> {fk.get('referenced_table')}.{fk.get('referenced_column')}\n"
        
        return description
    
    def _build_prompt(self, db_type: str, db_description: str, user_query: str) -> str:
        """构建提示语"""
        return f"""你是一位精通SQL的数据库专家。用户需要你根据下面的数据库结构，为用户的查询生成合适的SQL查询语句。

数据库类型: {db_type}

数据库结构:
{db_description}

用户查询: {user_query}

请生成一个高效、准确的SQL查询语句，满足用户的需求。只返回SQL语句本身，不需要任何解释或者markdown格式。确保生成的SQL符合指定数据库类型的语法规范。

SQL查询:"""
    
    def _extract_sql_from_response(self, response: str) -> str:
        """从大模型响应中提取SQL查询语句"""
        # 尝试去除可能的代码块标记
        if "```sql" in response:
            # 提取被```sql ```包围的内容
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        if "```" in response:
            # 提取被```包围的内容
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # 如果没有代码块标记，直接返回响应内容
        return response.strip() 