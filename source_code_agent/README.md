# CogmAIt 后端 API

这是CogmAIt项目的后端API，使用FastAPI框架开发，提供模型管理、知识管理和智能体管理等功能。本项目采用了可插拔式的模型提供商设计，支持灵活扩展不同的AI模型服务。

## 特性

- **可插拔式模型提供商**：每个新的AI模型提供商只需添加一个Python模块，系统会自动加载并集成
- **支持多种模型类型**：聊天、文本补全、嵌入向量等
- **RESTful API**：符合RESTful设计规范的API接口
- **自动文档生成**：基于FastAPI自动生成的API文档
- **异步处理**：使用异步编程提高性能
- **类型安全**：使用Pydantic进行数据验证和类型检查

## 安装

1. 克隆仓库

```bash
git clone <repository-url>
cd cogmait-backend
```

2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

1. 创建 `.env` 文件（可选）

```
# API设置
API_V1_STR=/api
PROJECT_NAME=CogmAIt

# 安全设置
SECRET_KEY=your-secret-key

# CORS设置
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# 数据库设置
DATABASE_URI=sqlite:///./cogmait.db

# 模型API密钥（可选）
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## 运行

```bash
python run.py
```

服务将在 http://localhost:8000 启动，API文档可在 http://localhost:8000/docs 访问。

## 目录结构

```
cogmait-backend/
├── app/                  # 应用程序目录
│   ├── api/              # API相关代码
│   │   └── v1/           # v1版本API
│   │       ├── api.py    # API路由聚合
│   │       └── endpoints/# API端点
│   ├── core/             # 核心配置
│   ├── db/               # 数据库相关
│   ├── models/           # 数据库模型
│   ├── providers/        # 模型提供商实现
│   │   ├── base.py       # 提供商基类
│   │   ├── manager.py    # 提供商管理器
│   │   ├── openai_provider.py # OpenAI实现
│   │   └── ...           # 其他提供商实现
│   ├── schemas/          # Pydantic模型
│   ├── utils/            # 工具函数
│   └── main.py           # 应用程序入口
├── requirements.txt      # 依赖项
├── run.py                # 运行脚本
└── README.md             # 项目说明
```

## 添加新的模型提供商

要添加新的模型提供商，只需在 `app/providers/` 目录中创建一个新的Python模块，并实现 `ModelProvider` 抽象基类：

1. 创建文件，例如 `app/providers/new_provider.py`
2. 实现 `ModelProvider` 基类的所有抽象方法
3. 重启应用程序，系统将自动加载新的提供商

示例：

```python
from app.providers.base import ModelProvider

class NewProvider(ModelProvider):
    @property
    def provider_id(self) -> str:
        return "new_provider"
    
    @property
    def provider_name(self) -> str:
        return "新提供商"
    
    # 实现其他必要方法...
```

## API文档

启动服务后，可以通过以下URL访问API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 许可

[MIT License](LICENSE)

# 知识图谱管理系统 - 后端

## Neo4j配置

1. 安装Neo4j数据库：从 [Neo4j官网](https://neo4j.com/download/) 下载并安装Neo4j Desktop或使用Docker

   ```bash
   # Docker安装示例
   docker run \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     -e NEO4J_PLUGINS=\[\"apoc\"\] \
     neo4j:5.15
   ```

2. 配置Neo4j连接信息：将`config.json.example`复制为`config.json`，并填写Neo4j连接信息

   ```bash
   cp config.json.example config.json
   # 然后编辑config.json文件
   ```

3. 安装Neo4j-GraphRAG所需的插件

   - APOC插件：提供高级功能
   - Vector插件：提供向量搜索功能

## 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Neo4j-GraphRAG集成

本系统使用Neo4j-GraphRAG来构建和查询知识图谱。主要功能包括：

1. 自动创建Neo4j子图：每个知识图谱对应一个Neo4j子图
2. 文本抽取和知识图谱构建：上传文件后，系统会解析文本并使用GraphRAG构建知识图谱
3. 图谱可视化：支持在前端展示Neo4j知识图谱

### 工作流程

1. 创建知识图谱：会自动创建Neo4j子图
2. 上传文件：系统解析文件内容
3. 知识提取：使用GraphRAG提取文本中的实体和关系
4. 可视化：在前端展示构建的知识图谱

### 环境变量配置（可选）

除了config.json，也可以使用环境变量配置Neo4j和OpenAI：

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=your_openai_key
``` 