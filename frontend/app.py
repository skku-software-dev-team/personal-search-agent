import os

import httpx
import pandas as pd
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

SEVERITY_EMOJI  = {"critical": "🔴", "medium": "🟡", "low": "🟢", "none": "⚪"}
SEVERITY_LABEL  = {"critical": "긴급",  "medium": "보통",  "low": "낮음"}
RELEVANCE_LABEL = {"high": "목표와 직결", "medium": "관련 있음", "low": "관련 낮음"}
FIELD_OPTIONS   = [
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
        if r.status_code == 200:
            st.success("🟢 백엔드 연결됨")
        else:
            st.error("🔴 백엔드 응답 오류")
    except Exception:
        st.error("🔴 백엔드 연결 안됨")


# ── 메인: 지식 공백 분석 ──────────────────────────────────────────────────────
st.title("🕳️ 지식 공백 분석")
st.caption(
    "문서를 KMeans로 클러스터링해 학습이 부족한 영역을 찾고, "
    "커리어 목표에 맞는 학습 로드맵을 제안합니다."
)

with st.expander("⚙️ 이번 분석만 다른 목표 사용하기", expanded=False):
    override_goal = st.text_input(
        "목표 override",
        placeholder="비워두면 사이드바 목표 자동 사용",
        label_visibility="collapsed",
    )

run = st.button("🔍 지식 공백 분석 시작", type="primary", use_container_width=True)

if not run:
    st.stop()

# ── 요청 본문 구성 ─────────────────────────────────────────────────────────────
if override_goal.strip():
    # override만 전송 → 나머지는 백엔드에서 저장된 프로필 사용
    request_body: dict = {"goal": override_goal.strip()}
elif goal.strip():
    request_body = {
        "goal": goal.strip(),
        "fields": fields,
        "level": level,
        "timeline": timeline,
    }
else:
    request_body = {}

# ── API 호출 ───────────────────────────────────────────────────────────────────
with st.spinner("클러스터링 & AI 분석 중... (10~30초)"):
    try:
        res = httpx.post(f"{BACKEND_URL}/gaps", json=request_body, timeout=120)
    except Exception as e:
        st.error(f"요청 실패: {e}")
        st.stop()

if res.status_code == 422:
    st.warning(f"⚠️ {res.json().get('detail', '문서가 부족합니다.')}")
    st.info("먼저 `POST /ingest/local` 또는 `POST /ingest`로 문서를 등록해주세요.")
    st.stop()
elif res.status_code != 200:
    st.error(f"오류 {res.status_code}: {res.text}")
    st.stop()

data      = res.json()
clusters  = data.get("clusters", [])
gaps      = data.get("gaps", [])
roadmap   = data.get("roadmap")
used_goal = data.get("goal")

# ── 분석 기준 배너 ─────────────────────────────────────────────────────────────
if used_goal:
    st.success(f"🎯 **{used_goal}** 목표 기준으로 분석했습니다.")
else:
    st.info(
        "📄 목표 없이 문서 기반으로만 분석했습니다. "
        "사이드바에서 커리어 목표를 설정하면 맞춤 추천과 로드맵을 받을 수 있어요."
    )

# ── 요약 지표 ──────────────────────────────────────────────────────────────────
gap_clusters = [c for c in clusters if c["is_gap"]]
critical_cnt = sum(1 for g in gaps if g.get("severity") == "critical")

m1, m2, m3, m4 = st.columns(4)
m1.metric("📄 총 청크",    data["total_chunks"])
m2.metric("🗂️ 클러스터",  data["n_clusters"])
m3.metric("🕳️ 공백 영역", len(gap_clusters))
m4.metric("🔴 긴급 공백", critical_cnt)

st.divider()

# ── 클러스터 분포 ──────────────────────────────────────────────────────────────
st.subheader("📊 클러스터별 문서 분포")

sorted_clusters = sorted(clusters, key=lambda x: x["doc_count"], reverse=True)

col_chart, col_table = st.columns([1, 1])

with col_chart:
    df_chart = pd.DataFrame([
        {
            "주제": (c["topic"][:14] + "…" if len(c["topic"]) > 14 else c["topic"]),
            "문서 수": c["doc_count"],
        }
        for c in sorted_clusters
    ])
    st.bar_chart(df_chart.set_index("주제"), color="#5470C6", height=300)

with col_table:
    rows = [
        {
            "주제":   c["topic"],
            "문서 수": c["doc_count"],
            "상태":   (
                f"{SEVERITY_EMOJI.get(c['severity'], '')} "
                f"{SEVERITY_LABEL.get(c['severity'], '정상') if c['severity'] != 'none' else '정상'}"
            ),
            "키워드": " · ".join(c["keywords"][:3]),
        }
        for c in sorted_clusters
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()

# ── 공백 추천 ──────────────────────────────────────────────────────────────────
st.subheader("🕳️ 지식 공백 & 추천")

if not gaps:
    st.balloons()
    st.success("공백이 발견되지 않았습니다! 지식이 고르게 분포되어 있어요. 🎉")
else:
    severity_rank = {"critical": 0, "medium": 1, "low": 2}
    for gap in sorted(gaps, key=lambda g: severity_rank.get(g.get("severity", "low"), 3)):
        sev = gap.get("severity", "medium")
        rel = gap.get("goal_relevance", "medium")

        with st.container(border=True):
            hc, bc = st.columns([4, 1])
            with hc:
                st.markdown(f"#### {SEVERITY_EMOJI.get(sev, '')} {gap['area']}")
            with bc:
                if used_goal:
                    st.caption(f"목표 연관성\n**{RELEVANCE_LABEL.get(rel, rel)}**")

            st.markdown(f"📌 {gap['reason']}")
            st.info(f"💡 {gap['recommendation']}")

            if gap.get("related_strong_topic"):
                st.caption(f"🔗 연관 강한 토픽: **{gap['related_strong_topic']}**")

# ── 학습 로드맵 ────────────────────────────────────────────────────────────────
if roadmap:
    st.divider()
    st.subheader(f"🗺️ 학습 로드맵 — {used_goal} ({timeline})")

    phase_cols = st.columns(len(roadmap))
    for phase, col in zip(roadmap, phase_cols):
        with col:
            with st.container(border=True):
                st.markdown(f"**Phase {phase['phase']}**")
                st.caption(f"📅 {phase['period']}")
                st.markdown(f"**{phase['focus']}**")
                st.markdown("---")
                for topic in phase.get("topics", []):
                    st.markdown(f"- {topic}")
