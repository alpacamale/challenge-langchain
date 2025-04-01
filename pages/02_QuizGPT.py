import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.retrievers import WikipediaRetriever
from pathlib import Path
import json
import os

page_title = "Quiz GPT"

st.set_page_config(
    page_title=page_title,
    page_icon="🤖",
)
st.title(page_title)


prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.

    Based Only on the following content make 5 questions to test the user's knowledge about the text.

    Each questions should have 4 answers, three of them must be incorrect and one should be correct.

    The difficulty level of the problem is '{level}'.

    Context: {context}
    """
)

create_quiz = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


@st.cache_data(show_spinner="Loading file ...")
def split_file(file):
    file_content = file.read()
    dir_path = "./.cache/quiz_files"
    file_path = f"{dir_path}/{file.name}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    return loader.load_and_split(text_splitter=splitter)


@st.cache_data(show_spinner="Searching ...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    return retriever.get_relevant_documents(term)


@st.cache_data(show_spinner="Making quiz ...")
def run_quiz_chain(_docs, topic, level):
    chain = prompt | llm
    return chain.invoke({"context": _docs, "level": level})


with st.sidebar:
    if os.environ.get("OPENAI_API_KEY") is None:
        api_key_input = st.text_input(
            "Enter your API key", type="password", key="api_key_input"
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.rerun()

    else:
        docs = None
        choice = st.selectbox(
            "Choose what you want to use.", ("Wikipedia Article", "File")
        )
        if choice == "File":
            file = st.file_uploader("Upload a .txt file")
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia")
            if topic:
                docs = wiki_search(topic)
        st.markdown("---")
        level = st.selectbox("Quiz level", ("EASY", "HARD"))
        st.markdown("---")

        st.link_button(
            label="Gihub Link",
            url="https://github.com/alpacamale/challenge-langchain",
        )

if os.environ.get("OPENAI_API_KEY") is not None:
    if docs is None:
        st.markdown(
            """

            Welcome to Quiz GPT

            I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

            Get started by uploading a file or searching on Wikipedia in the sidebar.
            """
        )
    else:
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
        ).bind(
            function_call={"name": "create_quiz"},
            functions=[create_quiz],
        )
        response = run_quiz_chain(docs, topic if topic else file.name, level)
        response = response.additional_kwargs["function_call"]["arguments"]
        response = json.loads(response)
        with st.form("questions_form"):
            all_pass = True
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if value:
                    if {"answer": value, "correct": True} in question["answers"]:
                        st.success("Correct!")
                    else:
                        st.error("Wrong!")
                        all_pass = False
            if all_pass:
                st.balloons()
            st.form_submit_button()


else:
    "#### Please input your OPENAI_API_KEY on the sidebar"
