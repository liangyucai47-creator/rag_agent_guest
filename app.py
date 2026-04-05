# -*- coding: utf-8 -*-
"""
RAG 智能客服 - Streamlit 前端

启动: cd rag-customer-service && source .venv/bin/activate && streamlit run app.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st

# 路径设置
sys.path.insert(0, str(Path(__file__).parent))

# 环境变量
os.environ.setdefault("EMBEDDING_USE_LOCAL", "true")

from engine import rag_chat, load_directory, get_stats, search_knowledge, add_text, get_chroma_client

# === 页面配置 ===
st.set_page_config(
    page_title="速游智能客服",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# === 自定义样式 ===
st.markdown("""
<style>
    .stApp { max-width: 900px; margin: 0 auto; }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 3rem;
    }
    .bot-msg {
        background: #f0f2f5;
        border-left: 4px solid #667eea;
        margin-right: 3rem;
    }
    .source-tag {
        display: inline-block;
        background: #e8eaf6;
        color: #3f51b5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .intent-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .intent-rag { background: #e3f2fd; color: #1976d2; }
    .intent-greeting { background: #e8f5e9; color: #388e3c; }
    .intent-complaint { background: #ffebee; color: #d32f2f; }
    .intent-faq { background: #fff3e0; color: #f57c00; }
</style>
""", unsafe_allow_html=True)

# === 缓存优化 ===
@st.cache_resource
def init_engine():
    return True

_init = init_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_loaded" not in st.session_state:
    st.session_state.knowledge_loaded = False


# === 侧边栏 ===
with st.sidebar:
    st.header("⚙️ 管理面板")

    # 知识库状态
    st.subheader("📚 知识库")
    if st.button("📊 查看统计", use_container_width=True):
        stats = get_stats()
        st.json(stats)

    # 加载知识库
    st.subheader("📂 加载文档")
    load_dir = st.text_input("文档目录", value="knowledge")
    if st.button("加载知识库", use_container_width=True, type="primary"):
        with st.spinner("正在加载..."):
            result = load_directory(load_dir)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"✅ 加载成功：{result['loaded_files']} 个文件，{result['total_chunks']} 个切片")
            st.session_state.knowledge_loaded = True

    # 添加知识
    st.subheader("✏️ 添加知识")
    new_text = st.text_area("输入内容", height=100, placeholder="输入要添加到知识库的内容...")
    if st.button("添加", use_container_width=True):
        if new_text.strip():
            result = add_text(new_text)
            st.success(f"✅ 添加成功：{result['chunks']} 个切片")
        else:
            st.warning("内容不能为空")

    # 检索测试
    st.subheader("🔍 检索测试")
    test_query = st.text_input("搜索关键词")
    if test_query and st.button("搜索", use_container_width=True):
        with st.spinner("检索中..."):
            results = search_knowledge(test_query, top_k=3)
        for r in results:
            st.markdown(f"**{r['source']}** (距离: {r['distance']:.4f})")
            st.caption(r['text'][:150] + "...")

    # 清空对话
    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("速游智能客服 v1.0")
    st.caption("RAG + bge-m3 + Kimi API")


# === 主页面 ===
st.title("🎮 速游智能客服")
st.caption("基于 RAG 检索增强生成，回答基于知识库内容")

# 意图图例
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<span class="intent-tag intent-rag">RAG 检索</span>', unsafe_allow_html=True)
with col2:
    st.markdown('<span class="intent-tag intent-greeting">问候</span>', unsafe_allow_html=True)
with col3:
    st.markdown('<span class="intent-tag intent-complaint">投诉</span>', unsafe_allow_html=True)
with col4:
    st.markdown('<span class="intent-tag intent-faq">常见问题</span>', unsafe_allow_html=True)

st.divider()

# 显示历史消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            sources_html = "<br>".join([f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"]])
        intent_class = f'intent-{msg.get("intent", "rag")}'
        intent_html = f'<span class="intent-tag {intent_class}">{msg.get("intent", "rag")}</span>'

        answer = msg["content"].replace("\n", "<br>")
        st.markdown(
            f'<div class="chat-message bot-msg">{intent_html} {sources_html}<br><br>{answer}</div>',
            unsafe_allow_html=True
        )

# 输入框
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-message user-msg">{prompt}</div>', unsafe_allow_html=True)

    # 调用 RAG
    with st.spinner("🤔 思考中..."):
        result = rag_chat(prompt)

    # 显示回答
    intent = result.get("intent", "rag")
    sources = result.get("sources", [])
    sources_html = ""
    if sources:
        sources_html = "<br>".join([f'<span class="source-tag">📄 {s}</span>' for s in sources])
    intent_class = f'intent-{intent}'
    intent_html = f'<span class="intent-tag {intent_class}">{intent}</span>'

    answer = result["answer"].replace("\n", "<br>")
    st.markdown(
        f'<div class="chat-message bot-msg">{intent_html} {sources_html}<br><br>{answer}</div>',
        unsafe_allow_html=True
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "intent": intent,
        "sources": sources,
    })

    # 显示元信息
    with st.expander("📊 调试信息"):
        st.json({
            "意图": intent,
            "置信度": result.get("confidence"),
            "来源": sources,
            "检索切片数": result.get("retrieved_chunks"),
            "错误": result.get("error"),
        })
