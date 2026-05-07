# RAG 问答系统 Web 测试

## 文件说明

- `rag_server.py` - FastAPI 后端服务器
- `rag_test.html` - 前端测试页面
- `rag_langgraph.py` - RAG 系统核心逻辑
- `config.json` - **统一配置文件**（端口、Milvus、Embedding/LLM、FAQ 路径等）
- `config.example.json` - 配置示例，复制为 `config.json` 后按需修改
- `config_loader.py` - 配置加载（支持环境变量覆盖）

## 配置说明

所有运行参数均在 `rag/config.json` 中配置，无需改代码。

| 配置块 | 说明 |
|--------|------|
| **server** | `host`、`port`（服务端口，默认 8001）、`reload` |
| **milvus** | `host`、`port`、`collection_name` |
| **embedding** | `api_key`、`model`（DashScope 文本向量） |
| **llm** | `api_key`、`model`、`base_url`、`temperature` |
| **faq** | `data_dir`（FAQ 目录，相对 EicAI_hfs）、`file_path`（单文件时） |
| **rag** | `candidates_k_default`、`summary_max_len` |
| **cors** | CORS 允许来源等 |

**敏感信息**：可在 `config.json` 中填写 `embedding.api_key`、`llm.api_key`，或使用环境变量 `RAG_EMBEDDING_API_KEY`、`RAG_LLM_API_KEY`（环境变量优先）。其他可选环境变量：`RAG_SERVER_PORT`、`RAG_SERVER_HOST`、`MILVUS_HOST`、`MILVUS_PORT`。

首次使用可复制 `config.example.json` 为 `config.json` 再修改。

## 启动服务器

### 方法 1: 直接运行 Python 脚本（端口等从 config.json 读取）

```bash
cd EicAI_hfs
python -m rag.rag_server
```

### 方法 2: 使用 uvicorn 命令

```bash
cd EicAI_hfs
uvicorn rag.rag_server:app --host 0.0.0.0 --port 8001 --reload
```

端口也可在 `config.json` 的 `server.port` 中修改；用方法 1 时会自动使用配置文件中的端口。

## 访问测试页面

服务器启动后，在浏览器中访问（端口以 config.json 中 `server.port` 为准，默认 8001）：

```
http://localhost:8001
```

## API 接口

### 1. 健康检查
```
GET /api/health
```

### 2. 查询接口
```
POST /api/query
Content-Type: application/json

{
    "query": "如何查询商机？"
}
```

响应格式：
```json
{
    "answer": "答案内容...",
    "tool_called": true,
    "retrieved_content": "检索到的内容...",
    "message_count": 3
}
```

## 注意事项

1. 首次启动时，系统会初始化 RAG 系统（加载数据、创建向量库），可能需要一些时间
2. 确保 Milvus 服务正在运行（192.168.40.197:19530）
3. 确保 FAQ Excel 文件存在于 `EicAI_hfs` 目录下

## 故障排查

- 如果页面无法加载，检查服务器是否正常启动
- 如果查询失败，查看服务器控制台的错误信息
- 确保所有依赖已安装：`pip install -r ../requirement.txt`
