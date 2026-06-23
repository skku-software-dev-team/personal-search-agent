import os

import httpx
import streamlit as st
from utils.auth import auth_headers, require_login

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

SOURCE_OPTIONS = {
    "전체": None,
    "로컬 파일": "local",
    "Google Drive": "gdrive",
    "Notion": "notion",
}
SCORE_COLOR = {(0.8, 1.0): "🟢", (0.6, 0.8): "🟡", (0.0, 0.6): "🔴"}

st.set_page_config(page_title="문서 검색", page_icon="🔍", layout="wide")
st.title("🔍 문서 검색")
st.caption("의미 기반 벡터 검색으로 관련 문서를 찾아줍니다.")
require_login()
with st.form("search_form"):
    col_q, col_k, col_src = st.columns([4, 1, 1])
    query = col_q.text_input("검색어", placeholder="예: 머신러닝 모델 배포 방법")
    top_k = col_k.number_input("최대 결과", min_value=1, max_value=50, value=10)
    source_label = col_src.selectbox("소스 필터", list(SOURCE_OPTIONS.keys()))
    submitted = st.form_submit_button(
        "🔍 검색", type="primary", use_container_width=True
    )

if not submitted:
    st.stop()

if not query.strip():
    st.warning("검색어를 입력해주세요.")
    st.stop()

with st.spinner("벡터 검색 중..."):
    try:
        params = {"q": query, "top_k": top_k}
        src = SOURCE_OPTIONS[source_label]
        if src:
            params["source"] = src
        res = httpx.get(
            f"{BACKEND_URL}/search", params=params, headers=auth_headers(), timeout=30
        )
    except Exception as e:
        st.error(f"요청 실패: {e}")
        st.stop()

if res.status_code == 422:
    st.warning(res.json().get("detail", "잘못된 요청"))
    st.stop()
elif res.status_code != 200:
    st.error(f"오류 {res.status_code}: {res.text}")
    st.stop()

data = res.json()
results = data.get("results", [])
total = data.get("total", 0)

st.markdown(f"**'{query}'** 검색 결과 — {total}건")

if not results:
    st.info("관련 문서를 찾지 못했습니다.")
    st.stop()

st.divider()

for i, r in enumerate(results):
    score = r.get("score", 0)
    dot = next((sym for (lo, hi), sym in SCORE_COLOR.items() if lo <= score < hi), "⚪")

    with st.container(border=True):
        col_info, col_score = st.columns([5, 1])
        with col_info:
            st.markdown(f"**{r['file_name']}**")
            st.caption(
                f"📁 {r['file_path']}  ·  소스: `{r['source']}`  ·  청크 #{r['chunk_index']}  ·  {r.get('created_at', '')}"
            )
        with col_score:
            st.metric("유사도", f"{score:.2f}", delta=None, label_visibility="visible")
            st.markdown(dot)

        with st.expander("내용 보기"):
            st.text(r.get("text", ""))
