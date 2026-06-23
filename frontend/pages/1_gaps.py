import os

import httpx
import pandas as pd
import streamlit as st
from utils.auth import auth_headers, require_login

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

SEVERITY_EMOJI = {"critical": "🔴", "medium": "🟡", "low": "🟢", "none": "⚪"}
SEVERITY_LABEL = {"critical": "긴급", "medium": "보통", "low": "낮음"}
RELEVANCE_LABEL = {"high": "목표와 직결", "medium": "관련 있음", "low": "관련 낮음"}
GAP_TYPE_LABEL = {
    "missing": "⬛ 자료 없음",
    "sparse": "📉 자료 부족",
    "review": "🔄 복습 추천",
}
FIELD_OPTIONS = [
    "MLOps",
    "LLM",
    "딥러닝",
    "머신러닝",
    "백엔드",
    "데이터 분석",
    "클라우드",
    "보안",
    "프론트엔드",
    "DevOps",
    "데이터베이스",
]
LEVELS = ["취준생", "주니어", "시니어"]
TIMELINES = ["3개월", "6개월", "1년", "2년"]

st.set_page_config(page_title="지식 공백 분석", page_icon="🕳️", layout="wide")
require_login()
# ── 최초 방문 시 저장된 프로필 자동 로드 ──────────────────────────────────────
if "profile_loaded" not in st.session_state:
    try:
        r = httpx.get(f"{BACKEND_URL}/user/profile", headers=auth_headers(), timeout=3)
        p = r.json() if r.status_code == 200 else {}
    except Exception:
        p = {}
    st.session_state["s_goal"] = p.get("goal") or ""
    st.session_state["s_fields"] = [
        f for f in (p.get("fields") or []) if f in FIELD_OPTIONS
    ]
    st.session_state["s_level"] = (
        p.get("level") if p.get("level") in LEVELS else "취준생"
    )
    st.session_state["s_timeline"] = (
        p.get("timeline") if p.get("timeline") in TIMELINES else "6개월"
    )
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
        payload = {
            "goal": goal or None,
            "fields": fields,
            "level": level,
            "timeline": timeline,
        }
        try:
            r = httpx.post(
                f"{BACKEND_URL}/user/profile",
                json=payload,
                headers=auth_headers(),
                timeout=5,
            )
            if r.status_code == 200:
                st.session_state.update(
                    {
                        "s_goal": goal,
                        "s_fields": fields,
                        "s_level": level,
                        "s_timeline": timeline,
                    }
                )
                st.success("저장됐어요!")
            else:
                st.error("저장 실패")
        except Exception as e:
            st.error(f"오류: {e}")

    if c2.button("🔄 불러오기", use_container_width=True):
        try:
            r = httpx.get(
                f"{BACKEND_URL}/user/profile", headers=auth_headers(), timeout=3
            )
            if r.status_code == 200:
                p = r.json()
                st.session_state.update(
                    {
                        "s_goal": p.get("goal") or "",
                        "s_fields": [
                            f for f in (p.get("fields") or []) if f in FIELD_OPTIONS
                        ],
                        "s_level": (
                            p.get("level") if p.get("level") in LEVELS else "취준생"
                        ),
                        "s_timeline": (
                            p.get("timeline")
                            if p.get("timeline") in TIMELINES
                            else "6개월"
                        ),
                    }
                )
                st.rerun()
        except Exception as e:
            st.error(f"오류: {e}")

    st.divider()
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=3)
        (
            st.success("🟢 백엔드 연결됨")
            if r.status_code == 200
            else st.error("🔴 백엔드 응답 오류")
        )
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
        res = httpx.post(
            f"{BACKEND_URL}/gaps",
            json=request_body,
            headers=auth_headers(),
            timeout=120,
        )
    except Exception as e:
        st.error(f"요청 실패: {e}")
        st.stop()

if res.status_code == 422:
    st.warning(f"⚠️ {res.json().get('detail', '문서가 부족합니다.')}")
    st.info("먼저 `POST /ingest/local` 또는 `POST /ingest`로 문서를 등록해주세요.")
    st.stop()
elif res.status_code == 401:
    st.error("🔑 OpenAI API 키가 올바르지 않습니다.")
    st.stop()
elif res.status_code == 429:
    detail = res.json().get("detail", "API 요청 한도 초과.")
    st.error(f"⏱️ {detail}")
    st.stop()
elif res.status_code == 502:
    st.error(f"🌐 {res.json().get('detail', 'OpenAI API 오류가 발생했습니다.')}")
    st.stop()
elif res.status_code != 200:
    st.error(f"오류 {res.status_code}: {res.text}")
    st.stop()

data = res.json()
clusters = data.get("clusters", [])
gaps = data.get("gaps", [])
roadmap = data.get("roadmap")
used_goal = data.get("goal")
required_areas = data.get("required_areas")

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
missing_cnt = sum(1 for g in gaps if g.get("gap_type") == "missing")
review_cnt = sum(1 for g in gaps if g.get("gap_type") == "review")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("📄 총 청크", data["total_chunks"])
m2.metric("🗂️ 클러스터", data["n_clusters"])
m3.metric("🕳️ 공백 영역", len(gaps))
m4.metric("⬛ 자료 없음", missing_cnt)
m5.metric("🔴 긴급 공백", critical_cnt)
m6.metric("🔄 복습 추천", review_cnt)

if required_areas:
    with st.expander(
        f"📋 목표 달성에 필요한 지식 영역 ({len(required_areas)}개)", expanded=False
    ):
        cols = st.columns(3)
        for i, area in enumerate(required_areas):
            cols[i % 3].markdown(f"- {area}")

st.divider()

# ── 클러스터 분포 ──────────────────────────────────────────────────────────────
st.subheader("📊 클러스터별 문서 분포")

sorted_clusters = sorted(clusters, key=lambda x: x["doc_count"], reverse=True)

col_chart, col_table = st.columns([1, 1])

with col_chart:
    df_chart = pd.DataFrame(
        [
            {
                "주제": (c["topic"][:14] + "…" if len(c["topic"]) > 14 else c["topic"]),
                "문서 수": c["doc_count"],
            }
            for c in sorted_clusters
        ]
    )
    st.bar_chart(df_chart.set_index("주제"), color="#5470C6", height=300)

with col_table:
    rows = [
        {
            "주제": c["topic"],
            "문서 수": c["doc_count"],
            "상태": (
                f"{SEVERITY_EMOJI.get(c['severity'], '')} "
                f"{SEVERITY_LABEL.get(c['severity'], '정상') if c['severity'] != 'none' else '정상'}"
            ),
            "최근성": (
                f"🔄 {c['avg_age_days'] // 30}개월 전"
                if c.get("avg_age_days", 0) >= 180
                else "✅ 최신"
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

    gaps_by_type = {"missing": [], "sparse": [], "review": []}
    for g in gaps:
        gaps_by_type.setdefault(g.get("gap_type", "missing"), []).append(g)
    for bucket in gaps_by_type.values():
        bucket.sort(key=lambda g: severity_rank.get(g.get("severity", "low"), 3))

    tab_defs = [
        ("missing", f"⬛ 자료없음  {len(gaps_by_type['missing'])}"),
        ("sparse", f"📉 자료부족  {len(gaps_by_type['sparse'])}"),
        ("review", f"🔄 복습추천  {len(gaps_by_type['review'])}"),
    ]
    active_tabs = [(key, label) for key, label in tab_defs if gaps_by_type[key]]

    tabs = st.tabs([label for _, label in active_tabs])

    for tab, (gap_type, _) in zip(tabs, active_tabs):
        with tab:
            for gap in gaps_by_type[gap_type]:
                sev = gap.get("severity", "medium")
                rel = gap.get("goal_relevance", "medium")
                expander_label = f"{SEVERITY_EMOJI.get(sev, '')} **{gap['area']}**" + (
                    f"  ·  {RELEVANCE_LABEL.get(rel, rel)}" if used_goal else ""
                )
                with st.expander(expander_label, expanded=False):
                    if gap_type == "review":
                        st.markdown(f"🕐 {gap['reason']}")
                        st.warning(f"💡 {gap['recommendation']}")
                    else:
                        st.markdown(f"📌 {gap['reason']}")
                        st.info(f"💡 {gap['recommendation']}")

                    if gap.get("related_strong_topic"):
                        st.caption(
                            f"🔗 연관 강한 토픽: **{gap['related_strong_topic']}**"
                        )

                    resources = gap.get("resources", [])
                    if resources:
                        st.markdown("**📚 추천 학습 자료**")
                        for r in resources:
                            st.markdown(f"- {r}")

# ── Agent 분석 로그 ────────────────────────────────────────────────────────────
agent_trace = data.get("agent_trace")
if agent_trace:
    with st.expander(
        f"🤖 Agent 분석 로그 ({len(agent_trace)}회 tool 호출)", expanded=False
    ):
        st.caption(
            "각 Agent가 ChromaDB에 직접 조회한 기록입니다. "
            "기존 방식(순차 LLM 호출)과 달리 Agent가 필요한 정보를 능동적으로 검색합니다."
        )
        agent_colors = {
            "KnowledgeMapAgent": "🗂️",
            "RequirementAgent": "🔍",
            "GapRecommendAgent": "💡",
        }
        for log in agent_trace:
            agent = log.get("agent", "")
            tool = log.get("tool", "")
            args = log.get("args", {})
            result = log.get("result", "")
            icon = agent_colors.get(agent, "🤖")
            st.markdown(
                f"{icon} **{agent}** → `{tool}` "
                f"({', '.join(f'{k}={v}' for k, v in args.items())})"
            )
            st.caption(f"↳ {result}")

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
