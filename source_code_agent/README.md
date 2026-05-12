# CogmAIt Backend

CogmAIt 是一个基于 FastAPI 的智能体后端服务，包含模型接入、知识库检索、图谱检索、MCP 工具编排与流式对话能力。


## 1. 技术栈与能力

- Python 3.11+
- FastAPI + Uvicorn
- SQLAlchemy + MySQL
- MinIO 对象存储
- Neo4j 图数据库（可选）
- 多模型 Provider 机制（OpenAI、Anthropic、Google、本地模型等）
- 流水线式对话编排：Audit -> Strategies -> Inference -> Filter

## 2. 目录总览

```text
source_code_agent/
├── app/
│   ├── api/                    # 路由层
│   ├── services/               # 业务编排与服务层
│   ├── providers/              # 模型提供商
│   ├── domain/                 # 领域对象与抽象数据类型
│   ├── db/                     # 数据访问与初始化
│   └── main.py                 # FastAPI 入口
├── scripts/
│   ├── bootstrap.ps1           # 首次初始化（依赖 + 基础设施）
│   ├── start-dev.ps1           # 日常开发启动
│   └── stop-dev.ps1            # 停止基础设施
├── tests/                      # 自动化测试
├── docker-compose.yml          # MySQL + MinIO
├── pyproject.toml              # Poetry 依赖与项目配置
├── run.py                      # 启动脚本（含 MCP 子进程）
└── README.md
```

## 3. 环境要求

请先安装以下工具。

- Docker Desktop（需可执行 `docker compose`）
- Python 3.11 或更高版本
- Poetry（建议最新版）

可选组件。

- Neo4j（如果需要图谱检索能力）
- 外部模型 API Key（OpenAI/Anthropic/Google 等）

## 4. 快速启动（Windows 推荐）

### 4.1 首次启动

在项目根目录执行。

```powershell
cd E:\source\source_code_agent
.\scripts\bootstrap.ps1
```

该脚本会执行以下动作。

- 启动 MySQL 容器
- 复用已有 `minio-server` 容器，避免容器名冲突
- 若无 MinIO 容器则自动创建
- 执行 `poetry install --no-root` 安装依赖

### 4.2 日常开发启动

```powershell
cd E:\source\source_code_agent
.\scripts\start-dev.ps1
```

该脚本会启动基础设施并运行后端服务。

### 4.3 停止服务

```powershell
cd E:\source\source_code_agent
.\scripts\stop-dev.ps1
```

## 5. 手动启动（跨平台）

如不使用 PowerShell 脚本，可手动执行。

```bash
docker compose up -d mysql minio
poetry install --no-root
poetry run python run.py
```

默认访问地址。

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- MinIO API: `http://127.0.0.1:9000`
- MinIO Console: `http://127.0.0.1:9001`

智能体类型枚举请统一使用 `GET /api/agents/types`（需登录）；流式对话接口在 OpenAPI 中声明为 `text/event-stream`，不再误标为 JSON 的 `AgentChatResponse`。

## 6. 配置说明

> **安全规则**：仓库只追踪 `*.example` 模板文件。`.env` 和 `config.json` 已被 `.gitignore` 排除，不要手动 `git add` 它们。

### 6.1 `.env`（必须）

```bash
cp .env.example .env
```

然后编辑 `.env`，填入以下关键值（其余可选项保留注释即可）：

| 变量 | 说明 |
|---|---|
| `SECRET_KEY` | JWT 签名密钥，建议用 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成 |
| `DB_PASSWORD` | MySQL root 密码，与 `docker-compose.yml` 中 `DB_PASSWORD` 保持一致 |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | MinIO 凭据，默认 `minioadmin` |
| `OPENAI_API_KEY` 等 | 至少配置一个模型提供商的 Key |
| `TAVILY_API_KEY` | 如需联网搜索则配置 |

### 6.2 `config.json`（Neo4j，可选）

```bash
cp config.json.example config.json
```

编辑 `config.json`，填入 Neo4j 连接信息。不使用图谱检索时可跳过此步骤。

也可以用环境变量替代 `config.json`（优先级更高）：

```bash
# 在 .env 中取消注释并填写
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j
```

## 7. 对话架构

当前对话主链路采用分阶段流水线，API 入口只负责调度。

1. **Audit**：请求合法性与访问上下文检查（`chat_pipeline._audit`）。
2. **Strategies**：策略注册表按 `is_active(agent)` 动态激活，符合开闭原则。
   - `strategies/web_search.py` — WebSearchStrategy（自包含逻辑）
   - `strategies/knowledge_retrieval.py` — KnowledgeRetrievalStrategy（自包含逻辑）
   - `strategies/graph.py` — GraphRetrievalStrategy（委托 GraphRetrievalService）
3. **Inference**：MCP 工具编排 + 模型流式推理（`mcp_service` + `model_inference_service`）。
4. **Filter**：结果收尾与聊天历史持久化（`chat_pipeline._run_filter`）。

**核心服务文件（每个职责独立）：**

```text
app/services/
├── chat_pipeline.py            # 四阶段编排器（主入口）
├── strategy_base.py            # BaseRetrievalStrategy + StrategyContext/Result
├── strategies/
│   ├── web_search.py           # 联网搜索策略
│   ├── knowledge_retrieval.py  # 知识库检索策略
│   └── graph.py                # 图谱检索策略
├── chat_events.py              # SSE 事件工厂（格式集中维护）
├── agent_access_service.py     # 访问解析（从 API 层抽离）
├── document_context_service.py
├── model_inference_service.py
├── mcp_service.py
├── chat_response_service.py
└── graph_retrieval_service.py
```

**新增策略只需：**
1. 创建 `strategies/your_strategy.py`，实现 `is_active()` 和 `execute()`
2. 在 `chat_pipeline.py` 的 `_strategy_registry` 列表里追加实例
3. 不需要修改任何现有策略或编排逻辑

## 8. 测试与质量检查

```bash
poetry run pytest
poetry run pytest tests/services -q
poetry run radon cc app/api/v1/endpoints/agents.py -s
```

说明。

- 推荐优先跑 `tests/services` 验证策略与流水线关键行为。
- 外部依赖不完整时可通过 mock/stub 保持核心测试可运行。

## 9. 常见问题

### 9.1 MinIO 容器名冲突

现象：`/minio-server is already in use`。  
处理：已在脚本内自动复用同名容器，通常无需手工处理。

### 9.2 Poetry 安装时报包路径错误

现象：`No file/folder found for package ...`。  
处理：使用 `poetry install --no-root`，项目脚本已内置该行为。

### 9.3 可选依赖缺失告警

部分功能（如向量检索、本地模型）依赖可选组件。未安装时相关能力会降级，不影响基础 API 启动。

## 10. 开发建议

- 新增增强能力时，优先通过策略接口接入，不要把分支逻辑堆回 API 入口。
- 变更对话主流程后，至少补充一条服务层自动化测试。
- 优先保持 `agents.py` 入口轻量，复杂逻辑下沉到 `services` 层。
