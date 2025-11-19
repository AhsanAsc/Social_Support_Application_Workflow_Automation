import streamlit as st

st.set_page_config(page_title="Applications", layout="wide")

st.title("Applications – Management")

st.info("This page will call the FastAPI `/applications` and `/documents` APIs.")

# TODO:
# - Login / token handling
# - Create application form
# - Upload documents per application
# - Trigger parsing & eligibility evaluation
