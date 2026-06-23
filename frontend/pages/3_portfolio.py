import os
from datetime import datetime

import requests
import streamlit as st
from utils.auth import require_login

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Portfolio", page_icon="👤", layout="wide")

st.title("👤 포트폴리오")
st.caption("ChromaDB에 저장된 문서를 기반으로 AI가 포트폴리오를 자동 생성합니다.")
require_login()

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    user_name = st.text_input("이름", value="개발자")
    max_docs = st.slider("분석할 문서 수", 10, 100, 50)

# 생성 버튼
if st.button("🚀 포트폴리오 생성", type="primary", use_container_width=True):
    with st.spinner("ChromaDB 검색 중 + AI 분석 중..."):
        try:
            res = requests.post(
                f"{BACKEND_URL}/portfolio/generate",
                json={"user_name": user_name, "max_docs": max_docs},
                timeout=60,
            )
            res.raise_for_status()
            data = res.json()
            st.session_state["portfolio"] = data
        except Exception as e:
            st.error(f"생성 실패: {e}")

# 포트폴리오 렌더링
if "portfolio" in st.session_state:
    p = st.session_state["portfolio"]

    # 헤드라인
    st.markdown(f"## 💬 _{p['headline']}_")
    st.markdown("---")

    # 소개
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📝 소개")
        st.write(p["summary"])

    with col2:
        st.subheader("🛠️ 기술 스택")
        skills_text = "  ".join([f"`{s}`" for s in p["skills"]])
        st.markdown(skills_text)

    st.markdown("---")

    # 타임라인 프로젝트 이력
    st.subheader("📅 프로젝트 타임라인")

    for i, proj in enumerate(p["projects"]):
        with st.container():
            col_date, col_content = st.columns([1, 4])

            with col_date:
                st.markdown(f"### {proj['date']}")

            with col_content:
                st.markdown(f"#### {proj['title']}")
                st.write(proj["description"])
                st.markdown(f"**역할:** {proj['role']}")
                tech_str = "  ".join([f"`{t}`" for t in proj["tech_stack"]])
                st.markdown(f"**기술:** {tech_str}")

            if i < len(p["projects"]) - 1:
                st.divider()
