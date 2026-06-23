import streamlit as st

BACKEND_URL = "http://localhost:8000"

# 이미 로그인된 경우 메인으로
if st.session_state.get("jwt_token"):
    st.switch_page("app.py")

st.title("🔍 Personal Search Agent")
st.markdown("---")
st.markdown("### Google 계정으로 로그인하세요")

if st.button("🔑 Google로 로그인", use_container_width=True):
    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={BACKEND_URL}/auth/login">',
        unsafe_allow_html=True,
    )
