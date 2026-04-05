# RAG 客服系统 - 高并发升级方案

## 架构总览

```
用户 → Nginx → FastAPI 集群 → Redis 缓存 → 向量库(Milvus) → LLM API
                  ↓                ↓
              请求队列          相似问题缓存
```

## 三大瓶颈

| 瓶颈 | 原因 | 解决方案 |
|------|------|----------|
| 向量检索 | ChromaDB 单进程、文件锁 | → Milvus（分布式，P99<50ms） |
| LLM API | 并发限制、429 | → Redis缓存 + 请求队列 + 多LLM负载均衡 |
| 服务端 | 单进程 Uvicorn | → 多 Worker + Nginx 负载均衡 |

## 分阶段升级路线

### 阶段 1：零成本（支撑 100 并发）
- `uvicorn --workers 4` 多进程
- Redis 缓存热门问题（语义相似度>0.95直接返回，覆盖60-80%）
- slowapi 限流（每用户10次/分钟）
- 意图分流（问候/投诉不走LLM）

### 阶段 2：中等投入（支撑 1000 并发）
- ChromaDB → Milvus（Docker部署，HNSW索引，元数据过滤）
- httpx 异步 Embedding（连接池复用）
- Celery + Redis 请求队列（削峰填谷）
- 热门问题预计算缓存

### 阶段 3：生产级（支撑万级并发）
- K8s + HPA 自动扩缩容
- 多 LLM 提供商（Kimi主 → 智谱/DeepSeek降级）
- 向量库读写分离
- 前端 CDN + SSE 推送

## 向量数据库选型

| 产品 | 架构 | 容量 | 延迟P99 | 分布式 | 价格 |
|------|------|------|---------|--------|------|
| Chroma | 嵌入式 | 百万级 | <200ms | ❌ | 免费 |
| Milvus | 分布式云原生 | 百亿级 | <50ms | ✅ | ~$2000/月(高配) |
| Qdrant | 开源/云托管 | 千万级 | <100ms | ✅ | 社区版免费 |
| Pinecone | 全托管Serverless | 十亿级 | <100ms | ✅ | $70/月起 |

## RAG 性能优化五要素

1. **分块策略**：按语义边界分块，RecursiveCharacterTextSplitter + overlap
2. **Embedding 模型**：中文推荐 BAAI/bge-zh-v1.5 或 m3e，与向量库保持一致
3. **向量库索引**：HNSW（高维低延迟），设置相似度阈值过滤噪声
4. **重排序**：交叉编码器(bge-reranker)对 top-k 二次打分
5. **缓存**：Redis 缓存高频问题，减少 LLM 调用

## 参考资料
- https://cloud.tencent.com/developer/article/2601251 （向量数据库选型指南）
- https://cloud.tencent.com/developer/article/2623302 （RAG性能调优实战）
- https://cloud.tencent.com/developer/article/2498700 （RAG流程优化终极指南）
