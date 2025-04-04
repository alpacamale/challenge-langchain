import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from pathlib import Path
import os


def parse_page(soup):
    header = soup.select_one("header")
    footer = soup.select_one("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading website ...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/vectorize\/).*",
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    # loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def get_retriever(docs):
    embeddings = OpenAIEmbeddings(api_key=st.session_state["api_key"])
    cache_dir = (
        Path(__file__).resolve().parent / "../.cache/site_embeddings"
    ).resolve()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_store = LocalFileStore(cache_dir)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_store)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5. 0 being not helpful to the user and 5 being helpful to the user.

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    
    Context: {context}
    Question: {question}
    """
)


def get_answers(docs, question) -> dict:
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources. Return the source as it is.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def condenser(answers, metadata):
    if metadata:
        return "\n\n".join(
            f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}"
            for answer in answers
        )
    return "\n\n".join(answer["answer"] for answer in answers)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = condenser(answers, True)
    return choose_chain.invoke({"question": question, "answers": condensed})


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "ai",
            "message": "HI! I am assistant ai for giving you information about cloudflare!",
            "expand_message": None,
        }
    ]


def add_history(role: str, message: str, expand_message=None) -> None:
    st.session_state["messages"].append(
        {"role": role, "message": message, "expand_message": expand_message}
    )


def display_message(role, message, expand_message=None):
    with st.chat_message(role):
        if expand_message:
            with st.expander("Scoring answers ..."):
                st.write(expand_message)
        st.write(message)


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        self.message = ""


title = "Cloudflare GPT"

st.set_page_config(
    page_title=title,
    page_icon="⛅",
)

st.title(title)
st.markdown(
    """
    Ask anything about this website's content — including:

    - **vectorize**
    - **ai-gateway**
    - **workers-ai**
    """
)


url = "https://developers.cloudflare.com/sitemap-index.xml"

with st.sidebar:
    if "api_key" not in st.session_state:
        api_key_input = st.text_input(
            "Enter your API key",
            type="password",
        )
        if api_key_input:
            st.session_state["api_key"] = api_key_input
            st.rerun()
    else:
        st.link_button(
            label="Gihub Link",
            url="https://github.com/alpacamale/challenge-langchain",
        )

if "api_key" not in st.session_state:
    st.warning("Please input your OPENAI_API_KEY")
else:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        api_key=st.session_state["api_key"],
    )
    docs = load_website(url)
    with st.spinner("Embedding documents ..."):
        retriever = get_retriever(docs)
    for message in st.session_state["messages"]:
        display_message(message["role"], message["message"], message["expand_message"])
    query = st.chat_input("Ask a question to the website")
    if query:
        display_message("human", query)
        add_history("human", query)
        with st.chat_message("ai"):
            with st.expander("Scoring answers ..."):
                answers = get_answers(retriever.invoke(query), query)
            result = choose_answer(answers)
            result = result.content.replace("$", "\$")
            add_history("ai", result, condenser(answers["answers"], False))
