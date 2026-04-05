# 人机协作（Human-in-the-loop）设计方案

## 架构总览

```
用户 → AI客服 → 判断是否转人工
                ↓ 是
         人工坐席队列（WebSocket房间）
                ↓
         人工坐席接单 → 直接和用户对话
                ↓
         AI 退到旁听（可辅助推荐答案）
```

## 会话状态机

```
ai_serving → waiting_human → human_serving → ended
                  ↑                │
                  └── AI超时 ──────┘
```

- **ai_serving**：AI 在服务
- **waiting_human**：用户要求转人工，排队等坐席
- **human_serving**：坐席已接手，AI 退到旁听
- **ended**：对话结束

## 核心组件

### 1. 消息路由

```python
class MessageRouter:
    def route(self, session_id, message):
        session = get_session(session_id)

        if session.status == "ai_serving":
            return ai_chat(message)

        elif session.status == "waiting_human":
            ai_reply = ai_chat(message)
            return f"{ai_reply}\n\n⏳ 人工客服排队中，预计等待{session.queue_position}人"

        elif session.status == "human_serving":
            # 转发给坐席
            push_to_agent(session.agent_id, {
                "session_id": session_id,
                "user_message": message,
                "ai_suggestion": ai_suggest(message)  # AI 旁听给建议
            })
            return None  # 等坐席回复
```

### 2. 坐席接单流程

```
用户说"转人工"
  → AI 回复"正在转接..."
  → 创建排队记录，存入 Redis Sorted Set
  → 坐席端 WebSocket 收到新排队通知
  → 坐席点击"接入" → session 状态变为 human_serving
  → 用户消息 ↔ WebSocket ↔ 坐席端
  → AI 在后台给建议（可选）
```

### 3. AI 旁听辅助

坐席对话时 AI 分析：
- 情绪检测：😐/😠/😢
- 意图分析：退款/技术故障/咨询
- 建议回复：一键采纳或手动输入
- 相关知识：自动检索知识库推荐

### 4. 坐席工作台界面

```
┌─────────────────────────────┐
│  速游客服工作台              │
│                             │
│  📋 等待接入 (3)            │
│  ├─ 用户A: 转人工-退款问题   │
│  ├─ 用户B: 转人工-技术故障   │
│  └─ 用户C: 转人工-账户问题   │
│                             │
│  💬 当前对话：用户A          │
│  ┌─────────────────────┐    │
│  │ 用户: 我想退款       │    │
│  │ [AI建议] 建议回复：...│    │
│  │ 坐席: 您好，请问订单号│    │
│  └─────────────────────┘    │
│                             │
│  [采纳AI建议] [手动输入]     │
└─────────────────────────────┘
```

## 技术实现要点

| 组件 | 技术 | 说明 |
|------|------|------|
| 会话管理 | Redis Hash | 存每个对话的状态、排队位置、坐席ID |
| 消息推送 | WebSocket 双向 | 用户和坐席都是长连接 |
| 排队队列 | Redis Sorted Set | 按时间排序，坐席接单时 pop |
| 坐席分配 | 轮询/最少对话数 | 自动分配给最闲的坐席 |
| AI 旁听 | 异步调用 | 不阻塞坐席对话 |
| 情绪检测 | LLM Prompt | 分析用户消息情绪 |
| 坐席认证 | JWT Token | 确保只有授权坐席可接入 |

## 待实现 TODO

- [ ] WebSocket 房间管理（用户-坐席配对）
- [ ] Redis 排队队列（Sorted Set）
- [ ] 会话状态机（4种状态流转）
- [ ] 坐席工作台前端（独立页面）
- [ ] AI 旁听推荐（异步不阻塞）
- [ ] 坐席认证系统（JWT）
- [ ] 坐席分配策略（轮询/负载均衡）
- [ ] 对话记录存储（MySQL/PostgreSQL）
- [ ] 满意度评价（对话结束后）
