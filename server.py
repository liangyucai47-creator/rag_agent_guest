# -*- coding: utf-8 -*-
"""
RAG 智能客服 - 生产级后端 v3.0

技术栈：FastAPI + ChromaDB + 智谱 Embedding API + Kimi LLM API + Redis
特性：流式输出、多轮对话、意图识别、知识库管理、WebSocket
高并发：Redis 语义缓存 + 请求限流 + 多 Worker 支持

启动: python3 server.py
多Worker: python3 server.py --workers 4
"""

import os
import re
import json
import hashlib
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import chromadb
from openai import OpenAI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis


# ============================================================
# 配置
# ============================================================

# LLM（Kimi）
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-ioLLNXeBiC6r7UjrbfN8b8MzacZ3VayvvqKj1PBZ7G1yQJg6")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.moonshot.cn/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "moonshot-v1-8k")

# Embedding（智谱）
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "7efb744022204110a0202b0a77794b72.IOYX4nJnHIYhT9Ow")
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embedding-3")

# Redis
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_TTL = int(os.environ.get("REDIS_CACHE_TTL", "3600"))  # 缓存1小时
REDIS_CACHE_PREFIX = "rag:cache:"

# 限流
RATE_LIMIT = os.environ.get("RATE_LIMIT", "20/minute")  # 每分钟20次

# 向量库
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
CHROMA_COLLECTION = "suyou_knowledge"

# 文档切片
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# 对话历史限制
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = """你是好运电商的智能客服助手。请根据以下参考信息回答用户问题。

参考信息：
{context}

规则：
1. 只基于参考信息回答，不编造
2. 不足时诚实说明
3. 简洁清晰，分点说明
4. 涉及退款、金额等敏感信息时提醒用户以实际页面为准"""

INTENT_KEYWORDS = {
    "greeting": ["你好", "在吗", "hello", "hi", "嗨", "您好"],
    "human_transfer": ["转人工", "人工客服", "真人", "找人工", "人工", "客服人员", "活人"],
    "complaint": ["投诉", "不满", "差评", "垃圾", "骗", "坑", "举报"],
    "refund": ["退款", "退货", "退换货", "退钱", "不想要了", "退款申请"],
    "logistics": ["物流", "快递", "发货", "到哪了", "运单号", "签收"],
}

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
UPLOAD_DIR = BASE_DIR / "knowledge" / "uploads"


# ============================================================
# Redis 缓存层
# ============================================================

class CacheLayer:
    """Redis 语义缓存：相似问题命中直接返回"""

    def __init__(self):
        self._redis = None
        self._enabled = False

    def connect(self):
        try:
            self._redis = redis.from_url(REDIS_URL, decode_responses=True)
            self._redis.ping()
            self._enabled = True
            print(f"[缓存] Redis 已连接: {REDIS_URL}")
        except Exception as e:
            self._enabled = False
            print(f"[缓存] Redis 不可用，降级为无缓存模式: {e}")

    @property
    def enabled(self):
        return self._enabled

    def _key(self, question: str) -> str:
        return REDIS_CACHE_PREFIX + hashlib.md5(question.strip().encode()).hexdigest()

    def get(self, question: str) -> Optional[Dict]:
        """查询缓存"""
        if not self._enabled:
            return None
        try:
            data = self._redis.get(self._key(question))
            if data:
                result = json.loads(data)
                result["from_cache"] = True
                print(f"[缓存] 命中: {question[:30]}...")
                return result
        except Exception:
            pass
        return None

    def set(self, question: str, result: Dict) -> None:
        """写入缓存"""
        if not self._enabled:
            return
        try:
            cache_data = {k: v for k, v in result.items() if k != "from_cache"}
            self._redis.setex(self._key(question), REDIS_CACHE_TTL, json.dumps(cache_data, ensure_ascii=False))
        except Exception:
            pass

    def clear(self) -> int:
        """清空缓存"""
        if not self._enabled:
            return 0
        try:
            keys = self._redis.keys(f"{REDIS_CACHE_PREFIX}*")
            if keys:
                return self._redis.delete(*keys)
        except Exception:
            pass
        return 0

    def stats(self) -> Dict:
        if not self._enabled:
            return {"enabled": False, "keys": 0}
        try:
            keys = self._redis.keys(f"{REDIS_CACHE_PREFIX}*")
            return {"enabled": True, "cached_questions": len(keys), "ttl_seconds": REDIS_CACHE_TTL}
        except Exception as e:
            return {"enabled": False, "error": str(e)}


# ============================================================
# Embedding 客户端
# ============================================================

class EmbeddingClient:
    """智谱 Embedding API 客户端"""

    def __init__(self):
        self._client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_API_BASE)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        all_emb = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_emb.extend([e.embedding for e in resp.data])
        return all_emb

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ============================================================
# 文档处理
# ============================================================

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """按段落 + 字数切片"""
    paragraphs = re.split(r'\n\n+', text)
    chunks, current = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current.strip())
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    c = para[i:i + chunk_size]
                    if c.strip():
                        chunks.append(c.strip())
                current = ""
            else:
                current = para
    if current:
        chunks.append(current.strip())
    return chunks


def load_file(filepath: str) -> str:
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix in ['.md', '.txt']:
        return path.read_text(encoding='utf-8')
    elif suffix == '.json':
        return json.dumps(json.loads(path.read_text(encoding='utf-8')), ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持: {suffix}")


# ============================================================
# 知识库
# ============================================================

class KnowledgeBase:
    """知识库管理（ChromaDB + API Embedding）"""

    def __init__(self):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._embed = EmbeddingClient()
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def load_directory(self, directory: str) -> Dict:
        dir_path = Path(directory)
        if not dir_path.exists():
            return {"error": f"目录不存在: {directory}"}

        files = [f for f in dir_path.rglob('*') if f.suffix.lower() in ['.md', '.txt', '.json']]
        if not files:
            return {"error": "目录中无支持的文件 (.md/.txt/.json)"}

        all_chunks, all_metas, all_ids = [], [], []
        for fp in files:
            try:
                text = load_file(str(fp))
                for i, chunk in enumerate(split_text(text)):
                    cid = hashlib.md5(f"{fp}:{i}".encode()).hexdigest()[:16]
                    all_chunks.append(chunk)
                    all_metas.append({"source": fp.name, "filepath": str(fp)})
                    all_ids.append(cid)
            except Exception as e:
                print(f"[警告] {fp}: {e}")

        if not all_chunks:
            return {"error": "没有有效内容"}

        embeddings = self._embed.embed(all_chunks)
        self.collection.add(documents=all_chunks, embeddings=embeddings, metadatas=all_metas, ids=all_ids)
        return {"loaded_files": len(files), "total_chunks": len(all_chunks)}

    def add_text(self, text: str, source: str = "manual") -> Dict:
        chunks = split_text(text)
        embeddings = self._embed.embed(chunks)
        ids = [hashlib.md5(f"{source}:{i}:{text[:50]}".encode()).hexdigest()[:16] for i in range(len(chunks))]
        self.collection.add(documents=chunks, embeddings=embeddings, metadatas=[{"source": source}]*len(chunks), ids=ids)
        return {"chunks": len(chunks)}

    def expand_query(self, query: str) -> str:
        """查询扩展：在同义词表中查找，补充关键词（不替换原问题）"""
        SYNONYMS = {
            "退款": ["退款", "退货", "退换货", "退钱", "退费"],
            "发货": ["发货", "快递", "物流", "配送", "包裹"],
            "订单": ["订单", "下单", "购买", "拍下", "交易"],
            "优惠券": ["优惠券", "券", "折扣券", "满减", "红包"],
            "支付": ["支付", "付款", "结算", "扣款"],
            "会员": ["会员", "VIP", "等级", "积分"],
            "商品": ["商品", "产品", "东西", "货"],
            "价格": ["价格", "多少钱", "价位", "贵不贵"],
            "怎么": ["怎么", "如何", "怎样"],
            "为什么": ["为什么", "原因", "怎么回事"],
            "到货": ["到货", "送达", "签收", "收到"],
            "售后": ["售后", "客服", "服务", "维修"],
        }
        extra = set()
        for word, syns in SYNONYMS.items():
            if word in query:
                extra.update(syns)
        if extra:
            extra.discard(query)  # 去掉可能重复的原词
            return query + " " + " ".join(list(extra)[:8])
        return query

    def search(self, query: str, top_k: int = 5, expand: bool = True) -> List[Dict]:
        """检索：支持查询扩展，扩大候选集后返回"""
        # 查询扩展
        expanded = self.expand_query(query) if expand else query
        retrieve_k = top_k * 3 if expand else top_k  # 扩展后多取候选

        q_emb = self._embed.embed_one(expanded)
        results = self.collection.query(query_embeddings=[q_emb], n_results=min(retrieve_k, self.collection.count() or 1))
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        items = [{"text": d, "source": m.get("source", ""), "distance": dist} for d, m, dist in zip(docs, metas, dists)]
        # 去重（同一文本可能被多次命中）
        seen = set()
        unique = []
        for item in items:
            key = item["text"][:50]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique[:retrieve_k]  # 返回候选集，交给 Reranker 精筛

    def stats(self) -> Dict:
        return {"total_chunks": self.collection.count(), "collection": CHROMA_COLLECTION}

    def reset(self) -> Dict:
        self._client.delete_collection(CHROMA_COLLECTION)
        self._collection = None
        return {"status": "ok"}


# ============================================================
# RAG 引擎（带缓存）
# ============================================================

class RAGEngine:
    """RAG 核心：缓存 → 意图识别 → 检索 → 生成"""

    def __init__(self, kb: KnowledgeBase, cache: CacheLayer):
        self.kb = kb
        self.cache = cache
        self._llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)

    def _contextual_query(self, question: str, history: List[Dict] = None) -> str:
        """上下文感知检索：结合对话历史重构查询（指代消解）"""
        if not history or len(history) < 2:
            return question

        # 只取最近几轮
        recent = history[-6:]
        history_text = "\n".join([f"{'用户' if m['role']=='user' else '客服'}: {m['content']}" for m in recent])

        prompt = f"""根据对话历史，将用户的最新问题改写为一个独立完整的问题。
保持原意，补充历史中提到的具体信息（如游戏名、平台等）。
只输出改写后的问题，不要解释。

对话历史：
{history_text}

最新问题：{question}

改写后的问题："""

        try:
            resp = self._llm.chat.completions.create(
                model=LLM_MODEL, temperature=0.0, max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            rewritten = resp.choices[0].message.content.strip()
            if rewritten and rewritten != question:
                print(f"[上下文感知] '{question}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            print(f"[上下文感知] 降级: {e}")
        return question

    def _self_evaluate(self, question: str, answer: str, sources: List[str]) -> Tuple[str, float]:
        """回答自检：评估回答是否基于文档、是否幻觉"""
        prompt = f"""评估这个AI回答的质量，只输出JSON。

用户问题：{question}
参考来源：{', '.join(sources) if sources else '无'}
AI回答：{answer[:500]}

评分标准：
- relevance: 回答是否切题 (0-1)
- grounded: 回答是否基于参考来源 (0-1)
- hallucination: 是否包含编造内容 (0-1, 0=无幻觉)

只输出JSON: {{"relevance": 0.8, "grounded": 0.9, "hallucination": 0.1}}"""

        try:
            resp = self._llm.chat.completions.create(
                model=LLM_MODEL, temperature=0.0, max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.choices[0].message.content.strip()
            # 提取JSON
            match = re.search(r'\{[^}]+\}', text)
            if match:
                scores = json.loads(match.group())
                relevance = scores.get("relevance", 0.5)
                grounded = scores.get("grounded", 0.5)
                hallucination = scores.get("hallucination", 0.5)
                score = (relevance + grounded - hallucination) / 2
                print(f"[自检] relevance={relevance:.1f} grounded={grounded:.1f} hallucination={hallucination:.1f} → 总分={score:.2f}")

                if score < 0.4 or hallucination > 0.6:
                    warning = "\n\n⚠️ 以上回答可能不够准确，建议联系人工客服确认。"
                    return answer + warning, score
                return answer, score
        except Exception as e:
            print(f"[自检] 降级: {e}")
        return answer, 0.5

    def _rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """用 LLM 对检索结果重排序（交叉编码器原理：逐对判断相关性）"""
        if len(results) <= top_k:
            return results[:top_k]

        # 构建打分 prompt
        docs_text = "\n".join([f"[{i}] {r['text']}" for i, r in enumerate(results)])
        prompt = f"""判断以下文档与问题的相关性，只返回最相关的{top_k}个文档编号，用逗号分隔。

问题：{query}

文档：
{docs_text}

最相关的{top_k}个编号："""

        try:
            resp = self._llm.chat.completions.create(
                model=LLM_MODEL, temperature=0.0, max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = resp.choices[0].message.content.strip()
            # 解析编号
            indices = []
            for part in re.split(r'[,，\s]+', answer):
                part = part.strip()
                match = re.search(r'\d+', part)
                if match:
                    idx = int(match.group())
                    if 0 <= idx < len(results):
                        indices.append(idx)
            # 去重保序
            seen = set()
            ranked = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    ranked.append(results[idx])
            # 补充未被选中的（保底）
            for r in results:
                if r not in ranked:
                    ranked.append(r)
            print(f"[Reranker] {len(results)} → {len(ranked[:top_k])} (prompt: {answer[:50]})")
            return ranked[:top_k]
        except Exception as e:
            print(f"[Reranker] 降级: {e}")
            return results[:top_k]

    def classify_intent(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        # 先匹配更具体的意图（避免"转人工客服"被 greeting 先匹配）
        priority_order = ["human_transfer", "refund", "logistics", "complaint", "greeting"]
        for intent in priority_order:
            keywords = INTENT_KEYWORDS.get(intent, [])
            if any(kw in text_lower for kw in keywords):
                return intent, 0.9
        return "rag", 0.5

    def _quick_reply(self, intent: str) -> Optional[str]:
        if intent == "greeting":
            return "您好！我是好运电商智能客服，请问有什么可以帮您的？\n\n常见问题：\n- 如何下单支付\n- 物流查询\n- 退换货流程\n- 优惠券使用\n- 会员权益"
        if intent == "human_transfer":
            return "好的，正在为您转接人工客服，请稍候...\n\n您也可以通过以下方式联系我们：\n- APP内「我的」→「客服中心」→「在线客服」\n- 客服热线：400-888-6666\n\n工作时间：周一至周日 9:00-22:00"
        if intent == "complaint":
            return "非常抱歉给您带来不好的体验，您的问题我已记录，客服人员会尽快联系处理。\n\n您也可以通过工单系统提交详细描述，我们会优先处理。"
        if intent == "refund":
            return "关于退换货：\n\n1. 支持7天无理由退换（食品/定制/已激活数码除外）\n2. 申请路径：我的订单 → 申请售后\n3. 退款到账时间：余额即时，原路1-3个工作日\n4. 如有质量问题，48小时内联系客服可免费退换\n\n请问您要退换的是哪个订单？"
        if intent == "logistics":
            return "物流查询方法：\n\n1. 打开APP → 「我的订单」→ 点击对应订单\n2. 可查看实时物流状态\n3. 支持复制运单号到快递官网查询\n\n请问您要查哪个订单的物流？"
        return None

    def chat(self, question: str, history: List[Dict] = None) -> Dict:
        # 1. 查缓存
        cached = self.cache.get(question)
        if cached:
            return cached

        intent, confidence = self.classify_intent(question)

        quick = self._quick_reply(intent)
        if quick:
            result = {"answer": quick, "intent": intent, "sources": [], "confidence": 1.0}
            self.cache.set(question, result)
            return result

        # 2. 上下文感知：结合历史重构查询
        retrieval_query = self._contextual_query(question, history)

        # 3. 检索（查询扩展 + Reranker）
        candidates = self.kb.search(retrieval_query, top_k=15, expand=True)
        results = self._rerank(retrieval_query, candidates, top_k=5)
        if not results:
            result = {"answer": "抱歉，知识库中暂未找到相关信息。建议联系人工客服。", "intent": "rag", "sources": [], "confidence": 0.0}
            self.cache.set(question, result)
            return result

        # 4. 构建上下文
        context_parts, sources = [], []
        for i, r in enumerate(results):
            if r["source"] and r["source"] not in sources:
                sources.append(r["source"])
            context_parts.append(f"[来源{i+1}: {r['source']}]\n{r['text']}")
        context = "\n\n".join(context_parts)

        # 5. LLM 生成
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            if history:
                messages.extend(history[-MAX_HISTORY_TURNS * 2:])
            messages.append({"role": "user", "content": question})

            resp = self._llm.chat.completions.create(
                model=LLM_MODEL, temperature=0.3, max_tokens=2048, messages=messages
            )
            answer = resp.choices[0].message.content

            # 6. 回答自检
            answer, eval_score = self._self_evaluate(question, answer, sources)

            result = {"answer": answer, "intent": "rag", "sources": sources, "confidence": eval_score}
        except Exception as e:
            fallback = "基于知识库检索：\n\n" + "\n\n".join([f"**{r['source']}**：{r['text'][:150]}..." for r in results[:3]])
            result = {"answer": fallback, "intent": "rag", "sources": sources, "confidence": 0.3, "error": str(e)}

        # 7. 写缓存
        self.cache.set(question, result)
        return result

    def chat_stream(self, question: str, history: List[Dict] = None):
        """流式生成"""
        # 缓存命中直接返回完整答案
        cached = self.cache.get(question)
        if cached:
            yield json.dumps({"type": "answer", "text": cached["answer"], "intent": cached.get("intent", "rag"), "sources": cached.get("sources", []), "from_cache": True}, ensure_ascii=False) + "\n"
            return

        intent, confidence = self.classify_intent(question)
        quick = self._quick_reply(intent)
        if quick:
            yield json.dumps({"type": "answer", "text": quick, "intent": intent, "sources": []}, ensure_ascii=False) + "\n"
            return

        results = self.kb.search(self._contextual_query(question, history), top_k=15, expand=True)
        results = self._rerank(question, results, top_k=5)
        if not results:
            yield json.dumps({"type": "answer", "text": "抱歉，知识库中暂未找到相关信息。", "intent": "rag", "sources": []}, ensure_ascii=False) + "\n"
            return

        context_parts, sources = [], []
        for i, r in enumerate(results):
            if r["source"] and r["source"] not in sources:
                sources.append(r["source"])
            context_parts.append(f"[来源{i+1}: {r['source']}]\n{r['text']}")
        context = "\n\n".join(context_parts)

        yield json.dumps({"type": "meta", "sources": sources, "intent": "rag"}, ensure_ascii=False) + "\n"

        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            if history:
                messages.extend(history[-MAX_HISTORY_TURNS * 2:])
            messages.append({"role": "user", "content": question})

            stream = self._llm.chat.completions.create(
                model=LLM_MODEL, temperature=0.3, max_tokens=2048, messages=messages, stream=True
            )
            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_answer += chunk.choices[0].delta.content
                    yield json.dumps({"type": "chunk", "text": chunk.choices[0].delta.content}, ensure_ascii=False) + "\n"

            # 流式完成后缓存
            self.cache.set(question, {"answer": full_answer, "intent": "rag", "sources": sources, "confidence": confidence})
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "text": str(e)}, ensure_ascii=False) + "\n"


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="RAG 智能客服 API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 限流
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 静态文件
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# 全局实例
cache = CacheLayer()
kb = KnowledgeBase()
rag = RAGEngine(kb, cache)


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict]] = None
    session_id: Optional[str] = None  # 会话ID


# ============================================================
# 会话管理（Redis 存储对话历史）
# ============================================================

class SessionManager:
    """管理用户会话和多轮对话历史"""

    SESSION_PREFIX = "rag:session:"
    SESSION_TTL = 86400 * 7  # 7天过期
    MAX_TURNS = 20  # 最多保留20轮

    def __init__(self, redis_client=None):
        self._redis = redis_client

    @property
    def enabled(self):
        return self._redis is not None

    def get_history(self, session_id: str) -> List[Dict]:
        if not self._redis:
            return []
        try:
            data = self._redis.get(f"{self.SESSION_PREFIX}{session_id}")
            return json.loads(data) if data else []
        except Exception:
            return []

    def append(self, session_id: str, role: str, content: str):
        if not self._redis:
            return
        try:
            history = self.get_history(session_id)
            history.append({"role": role, "content": content})
            # 只保留最近 MAX_TURNS 轮（每轮 = user + assistant）
            history = history[-(self.MAX_TURNS * 2):]
            self._redis.setex(
                f"{self.SESSION_PREFIX}{session_id}",
                self.SESSION_TTL,
                json.dumps(history, ensure_ascii=False)
            )
        except Exception:
            pass

    def clear(self, session_id: str):
        if not self._redis:
            return
        try:
            self._redis.delete(f"{self.SESSION_PREFIX}{session_id}")
        except Exception:
            pass

    def stats(self) -> Dict:
        if not self._redis:
            return {"enabled": False}
        try:
            keys = self._redis.keys(f"{self.SESSION_PREFIX}*")
            return {"enabled": True, "active_sessions": len(keys), "ttl_days": 7}
        except Exception:
            return {"enabled": False}


# 全局实例
session_mgr = SessionManager()


@app.on_event("startup")
def startup():
    cache.connect()
    if cache.enabled:
        session_mgr._redis = cache._redis
    if KNOWLEDGE_DIR.exists():
        result = kb.load_directory(str(KNOWLEDGE_DIR))
        if "error" not in result:
            print(f"[启动] 知识库已加载: {result['loaded_files']} 文件, {result['total_chunks']} 切片")


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rag-customer-service",
        "version": "4.0.0",
        "redis": cache.enabled,
        "cache_stats": cache.stats(),
        "knowledge": kb.stats(),
        "sessions": session_mgr.stats(),
    }


@app.post("/api/chat")
@limiter.limit(RATE_LIMIT)
def chat(req: ChatRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    # 加载会话历史
    history = req.history or []
    if req.session_id:
        history = session_mgr.get_history(req.session_id)
    result = rag.chat(req.question, history)
    # 保存对话
    if req.session_id:
        session_mgr.append(req.session_id, "user", req.question)
        session_mgr.append(req.session_id, "assistant", result.get("answer", ""))
    return result


@app.post("/api/chat/stream")
@limiter.limit(RATE_LIMIT)
async def chat_stream(req: ChatRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    # 加载会话历史
    history = req.history or []
    if req.session_id:
        history = session_mgr.get_history(req.session_id)

    # 记录用户消息
    if req.session_id:
        session_mgr.append(req.session_id, "user", req.question)

    # 收集完整回答用于存储
    full_answer = []
    sid = req.session_id

    def stream_with_save():
        for chunk in rag.chat_stream(req.question, history):
            yield chunk
            # 收集回答
            try:
                data = json.loads(chunk.strip())
                if data.get("type") == "chunk":
                    full_answer.append(data.get("text", ""))
                elif data.get("type") == "answer":
                    full_answer.append(data.get("text", ""))
            except Exception:
                pass
        # 保存 AI 回答
        if sid:
            session_mgr.append(sid, "assistant", "".join(full_answer))

    return StreamingResponse(
        stream_with_save(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    history = []
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            question = msg.get("question", "")
            if not question.strip():
                await ws.send_json({"type": "error", "text": "问题不能为空"})
                continue

            for chunk in rag.chat_stream(question, history):
                await ws.send_text(chunk)

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": ""})
    except WebSocketDisconnect:
        pass


# --- 知识库管理 ---

@app.get("/api/knowledge/stats")
def knowledge_stats():
    return kb.stats()


@app.post("/api/knowledge/load")
def load_knowledge(directory: str):
    if not os.path.isdir(directory):
        raise HTTPException(400, f"目录不存在: {directory}")
    result = kb.load_directory(directory)
    if "error" in result:
        raise HTTPException(400, result["error"])
    # 加载新知识后清空缓存
    cache.clear()
    return result


@app.post("/api/knowledge/add")
def add_knowledge(text: str, source: str = "manual"):
    result = kb.add_text(text, source)
    cache.clear()
    return result


@app.post("/api/knowledge/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / file.filename
    content = await file.read()
    path.write_bytes(content)
    return {"status": "ok", "file": file.filename, "path": str(path)}


@app.post("/api/knowledge/reset")
def reset_knowledge():
    result = kb.reset()
    cache.clear()
    return result


# --- 缓存管理 ---

@app.get("/api/cache/stats")
def cache_stats():
    return cache.stats()


@app.post("/api/cache/clear")
def cache_clear():
    count = cache.clear()
    return {"status": "ok", "cleared": count}


# --- 会话管理 ---

@app.post("/api/session/new")
def new_session():
    """创建新会话"""
    import uuid
    session_id = str(uuid.uuid4())[:8]
    return {"session_id": session_id}

@app.get("/api/session/{session_id}/history")
def get_history(session_id: str):
    """获取会话历史"""
    return {"session_id": session_id, "history": session_mgr.get_history(session_id)}

@app.post("/api/session/{session_id}/clear")
def clear_session(session_id: str):
    """清空会话历史"""
    session_mgr.clear(session_id)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="RAG 智能客服 v4.0")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8901, help="监听端口")
    parser.add_argument("--workers", type=int, default=1, help="Worker 进程数（多进程并发）")
    args = parser.parse_args()

    print("=" * 50)
    print("RAG 智能客服 v4.0")
    print(f"LLM: {LLM_MODEL} ({LLM_API_BASE})")
    print(f"Embedding: {EMBED_MODEL} ({EMBED_API_BASE})")
    print(f"Redis: {REDIS_URL}")
    print(f"限流: {RATE_LIMIT}")
    print(f"Workers: {args.workers}")
    print(f"知识库: {CHROMA_DIR}")
    print("=" * 50)

    if args.workers > 1:
        uvicorn.run("server:app", host=args.host, port=args.port, workers=args.workers)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
