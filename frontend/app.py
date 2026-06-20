import streamlit as st
import httpx
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Personal File Search Agent", page_icon="🧠")
st.title("🧠 Personal File Search Agent")
st.caption("내 문서를 AI로 검색·분석·추천")

try:
    res = httpx.get(f"{BACKEND_URL}/health", timeout=3)
    if res.status_code == 200:
        st.success("✅ Backend is connected")
except Exception as e:
    st.error(f"❌ Backend is not connected — {e}")