# CogmAIt Backend

CogmAIt 是一个基于 FastAPI 的智能体后端服务，包含模型接入、知识库检索、图谱检索、MCP 工具编排与流式对话能力。

本 README 作为项目唯一说明文档，目标是让新接手同学快速完成环境准备、启动运行与问题排查。

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

## 6. 配置说明

### 6.1 `.env`

复制 `.env.example` 为 `.env` 并按实际环境修改。

```bash
cp .env.example .env
```

常见配置包括数据库连接、鉴权密钥、跨域、模型 Key 等。

### 6.2 `config.json`

`run.py` 默认读取项目根目录 `config.json`，可通过环境变量 `CONFIG_FILE` 指向其他路径。

```bash
# Windows PowerShell 示例
$env:CONFIG_FILE = "E:\source\source_code_agent\config.json"
poetry run python run.py
```

## 7. 对话架构（重构后）

当前对话主链路采用分阶段流水线，API 入口只负责调度。

1. **Audit**：请求合法性与访问上下文检查。
2. **Strategies**：按智能体配置执行增强策略。
   - WebSearchStrategy
   - KnowledgeRetrievalStrategy
   - GraphRetrievalStrategy
3. **Inference**：调用模型流式推理并输出标准事件。
4. **Filter**：结果收尾与历史记录持久化。

对应核心文件。

- `app/api/v1/endpoints/agents.py`
- `app/services/chat_orchestration_service.py`
- `app/services/strategy_base.py`

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

## 11. 许可证

MIT License。