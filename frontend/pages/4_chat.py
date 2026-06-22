import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="채팅", page_icon="💬")
st.title("💬 내 문서와 대화하기")
st.caption("질문하면 관련 문서를 찾아 근거와 함께 답변해줘요.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 참고 문서 {len(msg['sources'])}개"):
                for s in msg["sources"]:
                    st.caption(f"[{s['source']}] {s['file_name']} — {s['file_path']}")

question = st.chat_input("내 문서에 대해 질문해보세요")
if question:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("검색하고 답변 작성 중..."):
            try:
                res = httpx.post(f"{BACKEND_URL}/chat", json={"question": question}, timeout=60)
                if res.status_code == 503:
                    answer, sources = "LLM API 키가 설정되지 않았습니다. `.env`에 OPENAI_API_KEY 또는 GROQ_API_KEY를 추가해주세요.", []
                elif res.status_code != 200:
                    answer, sources = f"오류 {res.status_code}: {res.text}", []
                else:
                    data = res.json()
                    answer, sources = data["answer"], data["sources"]
            except Exception as e:
                answer, sources = f"요청 실패: {e}", []

        st.markdown(answer)
        if sources:
            with st.expander(f"📎 참고 문서 {len(sources)}개"):
                for s in sources:
                    st.caption(f"[{s['source']}] {s['file_name']} — {s['file_path']}")

    st.session_state.chat_messages.append({"role": "assistant", "content": answer, "sources": sources})