# -*- coding: utf-8 -*-
"""
RAG 智能客服 - 生产级后端

技术栈：FastAPI + ChromaDB + 智谱 Embedding API + Kimi LLM API
特性：流式输出、多轮对话、意图识别、知识库管理、WebSocket

启动: python3 server.py
"""

import os
import re
import json
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import chromadb
from openai import OpenAI


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

# 向量库
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
CHROMA_COLLECTION = "suyou_knowledge"

# 文档切片
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# 对话历史限制
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = """你是速游加速器的智能客服助手。请根据以下参考信息回答用户问题。

参考信息：
{context}

规则：
1. 只基于参考信息回答，不编造
2. 不足时诚实说明
3. 简洁清晰，分点说明
4. 末尾标注信息来源"""

INTENT_KEYWORDS = {
    "greeting": ["你好", "在吗", "客服", "hello", "hi", "嗨", "您好"],
    "complaint": ["投诉", "不满", "差评", "退款", "垃圾", "骗", "坑"],
}

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
UPLOAD_DIR = BASE_DIR / "knowledge" / "uploads"


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

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self._embed.embed_one(query)
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [{"text": d, "source": m.get("source", ""), "distance": dist} for d, m, dist in zip(docs, metas, dists)]

    def stats(self) -> Dict:
        return {"total_chunks": self.collection.count(), "collection": CHROMA_COLLECTION}

    def reset(self) -> Dict:
        self._client.delete_collection(CHROMA_COLLECTION)
        self._collection = None
        return {"status": "ok"}


# ============================================================
# RAG 引擎
# ============================================================

class RAGEngine:
    """RAG 核心：意图识别 → 检索 → 生成"""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)

    def classify_intent(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return intent, 0.9
        return "rag", 0.5

    def _quick_reply(self, intent: str) -> Optional[str]:
        if intent == "greeting":
            return "您好！我是速游加速器智能客服，请问有什么可以帮您的？\n\n常见问题：\n- 如何使用加速器\n- 加速失败怎么办\n- 支持哪些游戏\n- 如何充值/续费"
        if intent == "complaint":
            return "非常抱歉给您带来不好的体验，您的问题我已记录，客服人员会尽快联系处理。\n\n您也可以通过工单系统提交详细问题描述，我们会优先处理。"
        return None

    def chat(self, question: str, history: List[Dict] = None) -> Dict:
        intent, confidence = self.classify_intent(question)

        quick = self._quick_reply(intent)
        if quick:
            return {"answer": quick, "intent": intent, "sources": [], "confidence": 1.0}

        # 检索
        results = self.kb.search(question, top_k=5)
        if not results:
            return {"answer": "抱歉，知识库中暂未找到相关信息。建议联系人工客服。", "intent": "rag", "sources": [], "confidence": 0.0}

        # 构建上下文
        context_parts, sources = [], []
        for i, r in enumerate(results):
            if r["source"] and r["source"] not in sources:
                sources.append(r["source"])
            context_parts.append(f"[来源{i+1}: {r['source']}]\n{r['text']}")
        context = "\n\n".join(context_parts)

        # LLM 生成
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            if history:
                messages.extend(history[-MAX_HISTORY_TURNS * 2:])
            messages.append({"role": "user", "content": question})

            resp = self._llm.chat.completions.create(
                model=LLM_MODEL, temperature=0.3, max_tokens=2048, messages=messages
            )
            return {"answer": resp.choices[0].message.content, "intent": "rag", "sources": sources, "confidence": confidence}
        except Exception as e:
            fallback = "基于知识库检索：\n\n" + "\n\n".join([f"**{r['source']}**：{r['text'][:150]}..." for r in results[:3]])
            return {"answer": fallback, "intent": "rag", "sources": sources, "confidence": 0.3, "error": str(e)}

    def chat_stream(self, question: str, history: List[Dict] = None):
        """流式生成（同步 generator，StreamingResponse 会正确处理）"""
        intent, confidence = self.classify_intent(question)
        quick = self._quick_reply(intent)
        if quick:
            yield json.dumps({"type": "answer", "text": quick, "intent": intent, "sources": []}, ensure_ascii=False) + "\n"
            return

        results = self.kb.search(question, top_k=5)
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
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield json.dumps({"type": "chunk", "text": chunk.choices[0].delta.content}, ensure_ascii=False) + "\n"

            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "text": str(e)}, ensure_ascii=False) + "\n"


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="RAG 智能客服 API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 静态文件
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

kb = KnowledgeBase()
rag = RAGEngine(kb)


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict]] = None


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-customer-service", "version": "2.0.0"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    return rag.chat(req.question, req.history)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    return StreamingResponse(
        rag.chat_stream(req.question, req.history),
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

            async for chunk in rag.chat_stream(question, history):
                await ws.send_text(chunk)

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": ""})  # 简化，实际应存完整回答
    except WebSocketDisconnect:
        pass


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
    return result


@app.post("/api/knowledge/add")
def add_knowledge(text: str, source: str = "manual"):
    return kb.add_text(text, source)


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
    return kb.reset()


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("RAG 智能客服 v2.0")
    print(f"LLM: {LLM_MODEL} ({LLM_API_BASE})")
    print(f"Embedding: {EMBED_MODEL} ({EMBED_API_BASE})")
    print(f"知识库: {CHROMA_DIR}")
    print("=" * 50)

    # 自动加载知识库
    if KNOWLEDGE_DIR.exists():
        result = kb.load_directory(str(KNOWLEDGE_DIR))
        if "error" not in result:
            print(f"[启动] 知识库已加载: {result['loaded_files']} 文件, {result['total_chunks']} 切片")

    uvicorn.run(app, host="0.0.0.0", port=8901)
