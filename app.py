import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from pathlib import Path
import os

page_title = "FullstackGPT Home"

st.set_page_config(
    page_title=page_title,
    page_icon="🤖",
)
st.title(page_title)

with st.sidebar:
    api_key = st.text_input("Write your api-key")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(temperature=0.1)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource
def get_memory():
    return ConversationBufferMemory(return_messages=True)


memory = get_memory()


def load_memory(_):
    return memory.load_memory_variables({})["history"]


cache_path = Path("./.cache").resolve()
cache_dir = LocalFileStore(cache_path)

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=100,
    separator="\n",
)

file = st.file_uploader(type="txt", label="Upload your document!")
if file is not None:
    with st.status("Implement RAG..."):
        st.write("Loading File...")
        with open(f"files/{file.name}", "w", encoding="utf-8") as f:
            f.write(file.read().decode("utf-8"))
        loader = TextLoader(f"files/{file.name}")
        st.write("Splitting into Chunks...")
        docs = loader.load_and_split(text_splitter=splitter)
        st.write("Embedding Chunks...")
        embedding = OpenAIEmbeddings()
        cached_embedding = CacheBackedEmbeddings.from_bytes_store(embedding, cache_dir)
        st.write("Save to Vectorstore...")
        vectorstore = FAISS.from_documents(docs, cached_embedding)
        retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer questions only using the following context. If you don't know the answer just say you don't know, don't make it up:\n{context}",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "history": RunnableLambda(load_memory),
        }
        | prompt
        | llm
    )

    def invoke_chain(question: str) -> str:
        result = chain.invoke(question)
        memory.save_context({"input": question}, {"output": result.content})
        return result.content

    def send_message(message, role):
        with st.chat_message(role):
            st.write(message)

    for message in st.session_state["messages"]:
        send_message(message["text"], message["role"])

    question = st.chat_input("Ask about your document")

    if question:
        send_message(question, "human")
        completion = invoke_chain(question)
        send_message(completion, "ai")
        st.session_state["messages"].append({"text": question, "role": "human"})
        st.session_state["messages"].append({"text": completion, "role": "ai"})
