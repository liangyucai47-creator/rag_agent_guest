# -*- coding: utf-8 -*-
"""
RAG 智能客服 - 核心引擎（轻量版，兼容 Python 3.9+）

Embedding 方案：优先用 OpenAI 兼容 API（便宜），fallback 用 ChromaDB 默认模型
"""

import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from openai import OpenAI


# === 配置 ===

LLM_CONFIG = {
    "api_key": os.environ.get("LLM_API_KEY", ""),
    "api_base": os.environ.get("LLM_API_BASE", "https://api.moonshot.cn/v1"),
    "model": os.environ.get("LLM_MODEL", "moonshot-v1-8k"),
    "temperature": 0.3,
    "max_tokens": 2048,
}

# Embedding 配置：用 OpenAI 兼容 API（DeepSeek/硅基流动等都支持）
EMBEDDING_CONFIG = {
    "api_key": os.environ.get("EMBEDDING_API_KEY", ""),
    "api_base": os.environ.get("EMBEDDING_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
    "model": os.environ.get("EMBEDDING_MODEL", "embedding-3"),
    "use_local": os.environ.get("EMBEDDING_USE_LOCAL", "false").lower() == "true",
    "local_model": "BAAI/bge-m3",  # 本地 Embedding 模型
    "persist_dir": os.path.join(os.path.dirname(__file__), "data", "chroma_db"),
}

CHROMA_COLLECTION = "suyou_knowledge"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

INTENT_KEYWORDS = {
    "faq": ["怎么", "如何", "为什么", "是什么", "吗", "？", "?", "教程", "使用方法"],
    "complaint": ["投诉", "不满", "差评", "退款", "垃圾", "骗", "坑"],
    "feedback": ["建议", "希望", "能不能", "如果", "功能"],
    "greeting": ["你好", "在吗", "客服", "hello", "hi", "嗨"],
}

SYSTEM_PROMPT = """你是速游加速器的智能客服助手。请根据以下参考信息回答用户问题。

参考信息：
{context}

回答要求：
1. 基于参考信息回答，不要编造
2. 如果参考信息不足以回答，请诚实说明
3. 回答简洁清晰，分点说明
4. 在回答末尾标注信息来源"""


# === 意图识别 ===

def classify_intent(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches >= 2:
            return intent, min(0.9, 0.5 + matches * 0.15)
    return "rag", 0.5


# === 文档处理 ===

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunk = para[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_file(filepath: str) -> str:
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix in ['.md', '.txt']:
        return path.read_text(encoding='utf-8')
    elif suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        return json.dumps(data, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


# === Embedding ===

_local_embed_model = None

def embed_texts(texts: List[str]) -> List[List[float]]:
    """文本向量化"""
    if EMBEDDING_CONFIG["use_local"]:
        return _embed_local(texts)
    else:
        return _embed_api(texts)


def _embed_api(texts: List[str]) -> List[List[float]]:
    """通过 API 向量化"""
    client = OpenAI(
        api_key=EMBEDDING_CONFIG["api_key"],
        base_url=EMBEDDING_CONFIG["api_base"],
    )
    # 批量处理，每次最多 64 条
    all_embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_CONFIG["model"],
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings


def _embed_local(texts: List[str]) -> List[List[float]]:
    """本地模型向量化（fallback）"""
    global _local_embed_model
    if _local_embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedding] 加载本地模型: {EMBEDDING_CONFIG['local_model']}")
        _local_embed_model = SentenceTransformer(EMBEDDING_CONFIG["local_model"])
    embeddings = _local_embed_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


# === 知识库管理 ===

def get_chroma_client() -> chromadb.ClientAPI:
    persist_dir = EMBEDDING_CONFIG["persist_dir"]
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(name: str = None) -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name or CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def load_directory(directory: str, collection_name: str = None) -> Dict:
    dir_path = Path(directory)
    if not dir_path.exists():
        return {"error": f"目录不存在: {directory}"}

    supported = ['.md', '.txt', '.json']
    files = [f for f in dir_path.rglob('*') if f.suffix.lower() in supported]
    if not files:
        return {"error": f"无支持的文件: {supported}"}

    collection = get_collection(collection_name)
    all_chunks, all_metadatas, all_ids = [], [], []

    for filepath in files:
        try:
            text = load_file(str(filepath))
            chunks = split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{filepath}:{i}".encode()).hexdigest()[:16]
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": filepath.name,
                    "filepath": str(filepath),
                    "chunk_index": i,
                })
                all_ids.append(chunk_id)
        except Exception as e:
            print(f"[警告] 加载失败 {filepath}: {e}")

    if not all_chunks:
        return {"error": "没有有效内容"}

    print(f"[知识库] 向量化 {len(all_chunks)} 个片段...")
    embeddings = embed_texts(all_chunks)

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    return {
        "loaded_files": len(files),
        "total_chunks": len(all_chunks),
        "collection": collection_name or CHROMA_COLLECTION
    }


def add_text(text: str, metadata: Dict = None, collection_name: str = None) -> Dict:
    collection = get_collection(collection_name)
    chunks = split_text(text)
    embeddings = embed_texts(chunks)
    ids = [hashlib.md5(f"manual:{i}:{text[:50]}".encode()).hexdigest()[:16] for i in range(len(chunks))]
    metas = [metadata or {} for _ in chunks]
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metas, ids=ids)
    return {"status": "ok", "chunks": len(chunks)}


def search_knowledge(query: str, top_k: int = 5, collection_name: str = None) -> List[Dict]:
    collection = get_collection(collection_name)
    query_embedding = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {"text": doc, "source": meta.get("source", "未知"), "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


# === RAG 生成 ===

def rag_chat(question: str, collection_name: str = None) -> Dict:
    intent, confidence = classify_intent(question)

    if intent == "greeting":
        return {
            "answer": "您好！我是速游加速器智能客服，请问有什么可以帮您的？\n\n常见问题：\n- 如何使用加速器\n- 加速失败怎么办\n- 支持哪些游戏\n- 如何充值/续费",
            "intent": "greeting", "sources": [], "confidence": 1.0
        }

    if intent == "complaint":
        return {
            "answer": "非常抱歉给您带来不好的体验，您的问题我已记录，客服人员会尽快联系处理。\n\n您也可以通过工单系统提交详细问题描述。",
            "intent": "complaint", "sources": [], "confidence": 0.85
        }

    results = search_knowledge(question, top_k=5, collection_name=collection_name)

    if not results:
        return {
            "answer": "抱歉，知识库中暂未找到相关信息。\n\n建议：\n1. 换种方式描述问题\n2. 联系人工客服",
            "intent": "rag", "sources": [], "confidence": 0.0
        }

    context_parts = []
    sources = []
    for i, r in enumerate(results):
        if r["source"] not in sources:
            sources.append(r["source"])
        context_parts.append(f"[来源{i+1}: {r['source']}]\n{r['text']}")

    context = "\n\n".join(context_parts)

    try:
        client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["api_base"])
        response = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": question},
            ]
        )
        return {
            "answer": response.choices[0].message.content,
            "intent": "rag", "sources": sources,
            "confidence": confidence, "retrieved_chunks": len(results)
        }
    except Exception as e:
        fallback = "基于知识库检索到以下相关信息：\n\n"
        for i, r in enumerate(results[:3]):
            fallback += f"**{r['source']}**：\n{r['text'][:200]}\n\n"
        fallback += "\n（LLM 暂不可用，以上为原始检索结果）"
        return {
            "answer": fallback, "intent": "rag", "sources": sources,
            "confidence": 0.3, "error": str(e)
        }


def get_stats(collection_name: str = None) -> Dict:
    collection = get_collection(collection_name)
    return {"total_chunks": collection.count(), "collection": collection_name or CHROMA_COLLECTION}
