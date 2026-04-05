# -*- coding: utf-8 -*-
"""
RAG 智能客服 - API 服务

启动: cd rag-customer-service && python3 api.py
"""

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from engine import rag_chat, load_directory, add_text, get_stats, search_knowledge

app = FastAPI(title="RAG 智能客服 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Models ===

class ChatRequest(BaseModel):
    question: str
    collection: Optional[str] = None

class AddDocRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None
    collection: Optional[str] = None


# === Routes ===

@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-customer-service"}


@app.post("/chat")
def chat(req: ChatRequest):
    """RAG 对话"""
    if not req.question.strip():
        raise HTTPException(400, detail="问题不能为空")
    return rag_chat(req.question, req.collection)


@app.get("/knowledge/stats")
def knowledge_stats(collection: Optional[str] = None):
    """知识库统计"""
    return get_stats(collection)


@app.post("/knowledge/load")
def load_knowledge(directory: str, collection: Optional[str] = None):
    """从目录加载文档到知识库"""
    if not os.path.isdir(directory):
        raise HTTPException(400, detail=f"目录不存在: {directory}")
    result = load_directory(directory, collection)
    if "error" in result:
        raise HTTPException(400, detail=result["error"])
    return result


@app.post("/knowledge/add")
def add_document(req: AddDocRequest):
    """添加单条文档"""
    return add_text(req.text, req.metadata, req.collection)


@app.post("/knowledge/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection: Optional[str] = None,
):
    """上传文件到知识库目录"""
    upload_dir = Path(__file__).parent / "knowledge" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return {"status": "ok", "file": file.filename, "path": str(file_path)}


@app.get("/knowledge/search")
def search(query: str, top_k: int = 5, collection: Optional[str] = None):
    """直接检索知识库（不经过 LLM）"""
    return {"results": search_knowledge(query, top_k, collection)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8901)
