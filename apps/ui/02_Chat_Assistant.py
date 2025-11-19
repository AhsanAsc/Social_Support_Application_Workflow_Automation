import streamlit as st

st.set_page_config(page_title="AI Chat Assistant", layout="wide")

st.title("AI Chat Assistant")

st.caption("Conversational interface over applications and documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about an application or upload documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # TODO: call FastAPI /chat endpoint
    assistant_reply = "Chat backend not implemented yet."
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
