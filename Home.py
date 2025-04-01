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

- [DocumentGPT](DocumentGPT)
- [QuizGPT](QuizGPT)
- [CloudflareGPT](CloudflareGPT)
"""
)
