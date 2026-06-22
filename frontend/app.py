import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

FIELD_OPTIONS = [
    "MLOps", "LLM", "딥러닝", "머신러닝", "백엔드", "데이터 분석",
    "클라우드", "보안", "프론트엔드", "DevOps", "데이터베이스",
]
LEVELS    = ["취준생", "주니어", "시니어"]
TIMELINES = ["3개월", "6개월", "1년", "2년"]

st.set_page_config(
    page_title="Personal Knowledge OS",
    page_icon="🧠",
    layout="wide",
)

# ── 최초 방문 시 저장된 프로필 자동 로드 ──────────────────────────────────────
if "profile_loaded" not in st.session_state:
    try:
        r = httpx.get(f"{BACKEND_URL}/user/profile", timeout=3)
        p = r.json() if r.status_code == 200 else {}
    except Exception:
        p = {}
    st.session_state["s_goal"]     = p.get("goal") or ""
    st.session_state["s_fields"]   = [f for f in (p.get("fields") or []) if f in FIELD_OPTIONS]
    st.session_state["s_level"]    = p.get("level") if p.get("level") in LEVELS else "취준생"
    st.session_state["s_timeline"] = p.get("timeline") if p.get("timeline") in TIMELINES else "6개월"
    st.session_state["profile_loaded"] = True

# ── 사이드바: 커리어 목표 ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎯 커리어 목표 설정")

    goal = st.text_input(
        "목표",
        value=st.session_state["s_goal"],
        placeholder="예: ML 엔지니어, 백엔드 개발자",
    )
    fields = st.multiselect(
        "관심 분야",
        options=FIELD_OPTIONS,
        default=st.session_state["s_fields"],
    )
    level = st.radio(
        "현재 레벨",
        LEVELS,
        index=LEVELS.index(st.session_state["s_level"]),
        horizontal=True,
    )
    timeline = st.select_slider(
        "목표 기간",
        options=TIMELINES,
        value=st.session_state["s_timeline"],
    )

    c1, c2 = st.columns(2)
    if c1.button("💾 저장", use_container_width=True, type="primary"):
        payload = {"goal": goal or None, "fields": fields, "level": level, "timeline": timeline}
        try:
            r = httpx.post(f"{BACKEND_URL}/user/profile", json=payload, timeout=5)
            if r.status_code == 200:
                st.session_state.update({
                    "s_goal": goal, "s_fields": fields,
                    "s_level": level, "s_timeline": timeline,
                })
                st.success("저장됐어요!")
            else:
                st.error("저장 실패")
        except Exception as e:
            st.error(f"오류: {e}")

    if c2.button("🔄 불러오기", use_container_width=True):
        try:
            r = httpx.get(f"{BACKEND_URL}/user/profile", timeout=3)
            if r.status_code == 200:
                p = r.json()
                st.session_state.update({
                    "s_goal":     p.get("goal") or "",
                    "s_fields":   [f for f in (p.get("fields") or []) if f in FIELD_OPTIONS],
                    "s_level":    p.get("level") if p.get("level") in LEVELS else "취준생",
                    "s_timeline": p.get("timeline") if p.get("timeline") in TIMELINES else "6개월",
                })
                st.rerun()
        except Exception as e:
            st.error(f"오류: {e}")

    st.divider()
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=3)
        st.success("🟢 백엔드 연결됨") if r.status_code == 200 else st.error("🔴 백엔드 응답 오류")
    except Exception:
        st.error("🔴 백엔드 연결 안됨")


# ── 메인: 홈 ──────────────────────────────────────────────────────────────────
st.title("🧠 Personal Knowledge OS")
st.markdown(
    "내가 공부한 문서들을 AI로 분석해 **지식 지도**를 만들고, "
    "**공백을 찾고**, **성장을 추적**합니다."
)

if st.session_state["s_goal"]:
    st.info(f"🎯 현재 목표: **{st.session_state['s_goal']}** ({st.session_state['s_level']} · {st.session_state['s_timeline']})")
else:
    st.warning("사이드바에서 커리어 목표를 설정하면 맞춤 분석을 받을 수 있어요.")

st.divider()

# ── 기능 카드 ─────────────────────────────────────────────────────────────────
st.subheader("기능")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 🕳️ 지식 공백 분석")
        st.markdown(
            "문서를 클러스터링해 내가 약한 영역을 찾고 "
            "커리어 목표에 맞는 학습 로드맵을 제안합니다."
        )
        st.page_link("pages/1_gaps.py", label="분석 시작 →", use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown("### 📈 지적 성장 타임라인")
        st.markdown(
            "기간별로 어떤 주제를 공부했는지 "
            "시간 흐름에 따라 시각화합니다."
        )
        st.page_link("pages/2_timeline.py", label="타임라인 보기 →", use_container_width=True)

with col3:
    with st.container(border=True):
        st.markdown("### 👤 포트폴리오 생성")
        st.markdown(
            "내 문서를 기반으로 AI가 자동으로 "
            "기술 스택과 프로젝트 이력을 정리합니다."
        )
        st.page_link("pages/3_portfolio.py", label="포트폴리오 만들기 →", use_container_width=True)

col4, col5 = st.columns(2)

with col4:
    with st.container(border=True):
        st.markdown("### 💬 문서와 대화")
        st.markdown(
            "내 문서를 RAG로 검색해 "
            "질문에 근거 기반으로 답변합니다."
        )
        st.page_link("pages/4_chat.py", label="채팅 시작 →", use_container_width=True)

with col5:
    with st.container(border=True):
        st.markdown("### 🔍 문서 검색")
        st.markdown(
            "키워드가 아닌 의미 기반 벡터 검색으로 "
            "관련 문서를 빠르게 찾습니다."
        )
        st.page_link("pages/5_search.py", label="검색하기 →", use_container_width=True)
