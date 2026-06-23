import os
from datetime import datetime, timedelta

import httpx
import plotly.graph_objects as go
import streamlit as st
from utils.auth import require_login

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

PERIOD_OPTIONS = {
    "1년": 365,
    "6개월": 180,
    "1개월": 30,
    "1주": 7,
    "1일": 1,
}

st.set_page_config(page_title="타임라인", page_icon="📈")
st.title("📈 지적 성장 타임라인")
require_login()
period = st.selectbox("기간 선택", list(PERIOD_OPTIONS.keys()))

if st.button("타임라인 생성"):
    days = PERIOD_OPTIONS[period]
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    with st.spinner("AI가 문서 분석 중..."):
        try:
            res = httpx.get(
                f"{BACKEND_URL}/timeline",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                timeout=60,
            )
            data = res.json()
            timeline = data.get("timeline", [])

            if not timeline:
                st.info("해당 기간에 문서가 없어요.")
            else:
                n = len(timeline)
                y = list(range(n - 1, -1, -1))  # 위에서 아래로

                fig = go.Figure()

                # 세로 연결선
                fig.add_trace(
                    go.Scatter(
                        x=[0] * n,
                        y=y,
                        mode="lines",
                        line=dict(color="#CBD5E1", width=2),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

                # 동그라미 노드
                fig.add_trace(
                    go.Scatter(
                        x=[0] * n,
                        y=y,
                        mode="markers",
                        marker=dict(
                            size=18, color="#4F8EF7", line=dict(color="white", width=2)
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

                # 키워드 + 한줄요약
                for i, item in enumerate(timeline):
                    yi = y[i]
                    # 키워드 (크게)
                    fig.add_annotation(
                        x=0.08,
                        y=yi + 0.15,
                        text=f"<b>{item.get('keyword', '')}</b>",
                        showarrow=False,
                        font=dict(size=15, color="#1a1a1a"),
                        xanchor="left",
                        xref="paper",
                    )
                    # 한줄요약 (작게)
                    fig.add_annotation(
                        x=0.08,
                        y=yi - 0.15,
                        text=f"<span style='color:#888'>{item.get('summary', '')}</span>",
                        showarrow=False,
                        font=dict(size=11, color="#888"),
                        xanchor="left",
                        xref="paper",
                    )

                fig.update_layout(
                    height=max(120 * n, 300),
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(visible=False, range=[-0.1, 1]),
                    yaxis=dict(visible=False, range=[-0.8, n - 0.2]),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"에러: {e}")
