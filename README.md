# Nomad Coder LangChain Challenge

LangChain Challenge for improve skills

## Features

### [Text completion using FewShotTemplate](notebooks/fewshot.ipynb)

Generates detailed movie information, including:

- **Director**
- **Starring**
- **Budget**
- **Box office revenue**
- **Genre**
- **Synopsis**

### [Implementing Completion with LCEL and Memory](notebooks/memory.ipynb)

Provides AI-powered responses based on LCEL and memory

- The AI responds with exactly **three emojis** representing the given movie title.
- It remembers previous conversations using a memory buffer.

### [Implementing RAG with LCEL and Memory](notebooks/rag.ipynb)

Provides AI-powered responses based on RAG

- The AI responds strictly based on the context retrieved from a document.
- It maintains conversation history using a buffer memory.
