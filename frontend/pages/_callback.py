import streamlit as st

params = st.query_params

if "token" in params:
    st.session_state["jwt_token"] = params["token"]
    st.query_params.clear()
    st.switch_page("app.py")
else:
    st.error("로그인 실패. 다시 시도해주세요.")
    st.page_link("pages/_login.py", label="로그인 페이지로 돌아가기")
