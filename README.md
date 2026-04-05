# RAG 智能客服系统 - 后端 API 文档

> 基于检索增强生成（RAG）的智能客服后端，支持多轮对话、意图识别、人工转接、AI 旁听辅助。

## 技术栈

- **Web 框架**: FastAPI
- **向量数据库**: ChromaDB
- **Embedding**: 智谱 API (embedding-3)
- **LLM**: Kimi/Moonshot API (moonshot-v1-8k)
- **缓存/会话**: Redis
- **限流**: slowapi

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 必填 - LLM 和 Embedding API 密钥
export LLM_API_KEY="your_kimi_api_key"
export EMBED_API_KEY="your_zhipu_api_key"

# 可选 - 坐席工作台密码（默认 changeme）
export AGENT_PASSWORD="your_password"

# 可选 - Redis 地址（默认 redis://localhost:6379/0）
export REDIS_URL="redis://localhost:6379/0"
```

### 3. 准备知识库

将 Markdown 文件放入 `knowledge/` 目录，启动时自动加载。当前包含：
- `user_guide.md` — 用户使用指南
- `faq.md` — 常见问题

### 4. 启动服务

```bash
# 单进程（开发）
python3 server.py --port 8901

# 多进程（生产，4 Worker）
python3 server.py --workers 4 --port 8901
```

启动后访问：
- `http://localhost:8901` — 用户聊天页面（测试用）
- `http://localhost:8901/agent` — 坐席工作台

## API 接口

### 对话接口

#### 单次对话
```
POST /api/chat
Content-Type: application/json

{
  "question": "怎么退换货？",
  "session_id": "abc123",    // 可选，用于多轮对话
  "history": []               // 可选，手动传入历史
}

响应:
{
  "answer": "关于退换货：...",
  "intent": "refund",
  "sources": [{"source": "faq.md", "text": "...", "score": 0.85}],
  "confidence": 0.9
}
```

#### 流式对话（推荐）
```
POST /api/chat/stream
Content-Type: application/json

请求体同上

响应: NDJSON 流（每行一个 JSON）
{"type": "meta", "intent": "rag", "sources": [...]}
{"type": "chunk", "text": "关于"}
{"type": "chunk", "text": "退换货"}
{"type": "answer", "text": "关于退换货...", "sources": [...], "intent": "rag"}
```

**注意**：返回的是 NDJSON 格式（不是 SSE），按 `\n` 分行解析。

### 意图类型

| intent | 触发关键词 | 行为 |
|--------|-----------|------|
| `greeting` | 你好、hello、hi | 快捷回复问候语 |
| `human_transfer` | 转人工、人工客服 | 快捷回复 + 触发转接流程 |
| `refund` | 退款、退货、退换货 | 快捷回复退货流程 |
| `logistics` | 物流、快递、发货 | 快捷回复物流查询方法 |
| `complaint` | 投诉、举报 | 快捷回复投诉处理 |
| `rag` | 其他所有 | RAG 检索 + LLM 生成 |

### 会话管理

```
POST /api/session/new              → 创建新会话，返回 session_id
GET  /api/session/{id}/history     → 获取会话历史
POST /api/session/{id}/clear       → 清空会话历史
```

### 转人工

```
POST /api/human/transfer?session_id=xxx  → 用户进入排队
GET  /api/human/queue                     → 查看排队状态
```

### WebSocket 接口

#### 用户 WebSocket（人工模式）
```
ws://host/ws/user/{session_id}

发送: {"text": "我的快递到哪了"}
接收: {"type": "waiting", "queue_position": 2}
      {"type": "agent_joined", "agent_id": "abc12345"}
      {"type": "agent_message", "text": "您好，我帮您查一下"}
      {"type": "ai_suggestion", "suggestion": "建议回复..."}  // AI 旁听建议
      {"type": "agent_left", "text": "服务结束"}
```

#### 坐席 WebSocket
```
ws://host/ws/agent/{agent_id}

发送:
  {"action": "accept", "session_id": "xxx"}     // 接入排队用户
  {"action": "message", "session_id": "xxx", "text": "回复内容"}  // 发消息给用户
  {"action": "end", "session_id": "xxx"}        // 结束服务

接收:
  {"type": "online", "agent_id": "xxx", "queue_count": 0}
  {"type": "new_queue", "session_id": "xxx", "queue_count": 1}
  {"type": "user_message", "session_id": "xxx", "text": "用户消息"}
  {"type": "ai_suggestion", "session_id": "xxx", "suggestion": "AI 建议回复"}
  {"type": "session_ended", "session_id": "xxx"}
```

#### 坐席登录
```
POST /api/agent/login?password=xxx
→ {"agent_id": "abc12345", "status": "ok"}
```

### 知识库管理

```
GET  /api/knowledge/stats          → 知识库统计（文档数、切片数）
POST /api/knowledge/load           → 加载指定目录的知识库
POST /api/knowledge/add            → 手动添加知识文本
POST /api/knowledge/upload         → 上传文件
POST /api/knowledge/reset          → 重置知识库
```

### 缓存管理

```
GET  /api/cache/stats              → 缓存统计
POST /api/cache/clear              → 清空缓存
```

## RAG 处理流程

```
用户问题
  ↓
Redis 语义缓存（命中则直接返回）
  ↓
意图识别（关键词匹配 + 短文本模糊兜底）
  ↓
快捷回复（refund/logistics/greeting 等）
  ↓
上下文感知（结合多轮历史重构查询）
  ↓
查询扩展（同义词补充）
  ↓
ChromaDB 向量检索（Top-15 候选）
  ↓
LLM 重排序（15 → Top-5）
  ↓
Kimi LLM 生成回答
  ↓
自检评分（相关性/事实性/幻觉检测）
  ↓
缓存结果 + 存储会话历史
```

## 转人工流程

```
用户触发"转人工"（关键词 / 按钮）
  ↓
后端创建排队记录，状态: waiting_human
  ↓
通知所有在线坐席（WebSocket）
  ↓
坐席点击"接入" → 状态: human_serving
  ↓
用户 ↔ 坐席 WebSocket 实时对话
  ↓
AI 旁听：用户发消息后异步生成回复建议
  ↓
坐席点"结束" → 状态: ended
```

## 对接自己的前端

后端是纯 API 服务，可对接任意前端（Vue/React/Flutter 等）。

**最小对接步骤：**

1. 调用 `POST /api/session/new` 获取 session_id
2. 用户发消息时调 `POST /api/chat/stream`，按 `\n` 分行解析 NDJSON
3. 收到 `intent: human_transfer` 时，调 `POST /api/human/transfer`
4. 转接后连 `ws://host/ws/user/{session_id}` 收发消息

CORS 已默认允许所有来源，无需后端额外配置。

## 性能测试

内置测试页面可作为前端压力测试：

```bash
# 启动服务
python3 server.py --port 8901

# 访问聊天页面测试对话
open http://localhost:8901

# 并发测试示例（用 ab 或 wrk）
ab -n 100 -c 20 -p test.json -T application/json \
   http://localhost:8901/api/chat

# WebSocket 测试可用 websocat
websocat ws://localhost:8901/ws/user/test_session
```

**已知性能数据：**
- 缓存命中: ~0.02s
- 缓存未命中: ~4.3s（20 并发最差）
- Embedding: ~0.7s（API 调用）
- LLM 生成: ~3s

## 环境变量汇总

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `LLM_API_KEY` | ✅ | Kimi/Moonshot API 密钥 | 无 |
| `EMBED_API_KEY` | ✅ | 智谱 Embedding API 密钥 | 无 |
| `AGENT_PASSWORD` | ❌ | 坐席工作台登录密码 | changeme |
| `REDIS_URL` | ❌ | Redis 连接地址 | redis://localhost:6379/0 |
| `RATE_LIMIT` | ❌ | 限流（请求/分钟） | 20/minute |
