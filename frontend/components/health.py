import streamlit as st
import httpx


def render_health(backend_url: str):
    try:
        res = httpx.get(f"{backend_url}/health", timeout=3)
        if res.status_code == 200:
            st.success("✅ Backend is connected")
    except Exception as e:
        st.error(f"❌ Backend is not connected — {e}")