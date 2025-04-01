import streamlit as st

page_title = "FullstackGPT Home"

st.set_page_config(
    page_title=page_title,
    page_icon="🤖",
)
st.title(page_title)

st.markdown(
    """
#### Links

- [DocumentGPT](pages/01_DocumentGPT.py)
- [QuizGPT](pages/02_QuizGPT.py)
"""
)
