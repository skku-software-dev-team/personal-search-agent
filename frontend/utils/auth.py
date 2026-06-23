import streamlit as st


def require_login():
    if "jwt_token" not in st.session_state or not st.session_state["jwt_token"]:
        st.switch_page("pages/_login.py")
        st.stop()


def get_token() -> str:
    return st.session_state.get("jwt_token", "")


def auth_headers() -> dict:
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def logout():
    st.session_state.clear()
    st.switch_page("pages/_login.py")
