# Hybrid LangChain RAG Implementation for a ChatBot 

**********************************************
Software:	&emsp;	Python 3.12

Version:	&emsp;  1.0

Date: 	&emsp;		Nov 29, 2025

Author:	&emsp;		Dirk Mueller
**********************************************

## Purpose  
This repository contains a small Retrieval-Augmented Generation (RAG) demo that lets you ask natural‑language questions about local regulatory documents (for example Medical Device Regulation or FDA guidance documents) from the command line.  
Markdown documents are indexed into a vector store, and GPT‑4o is used to answer questions grounded in those documents.  
## High‑level architecture
Conceptually, the flow is:  
Markdown files + (optional) images  
│  
▼
Text / image loader (LangChain)  
│  
▼
MarkdownTextSplitter → chunks  
│  
▼
OpenAI embeddings (text-embedding-3-large)  
│  
▼
FAISS vector store on disk  
│  
▼
Retriever (similarity search on question)  
│  
▼
Prompt + GPT‑4o (LangChain Runnables)  
│  
▼
Answer printed in the terminal  
## Components  
LangChain RAG with a vector database, implemented as Hybrid LLM has the following components:  
1. Python Version 3.12  
2. langchain>=1.1.0  
3. langchain-core  
4. langchain-community  
5. langchain-openai  
6. faiss-cpu  
7. langchain-text-splitters  
8. python-dotenv  
9. pillow

## Implementation

The implementation is a “modern” LangChain 1.x RAG pipeline with (LangChain Expression Language) LCEL runnables. LCEL is the “new style” way in LangChain to build chains using small, composable blocks called runnables, connected with the pipe operator |. and a local FAISS vector store:

- **Python 3.12** – runtime and virtual environment  
- **LangChain 1.x stack**  
  - `langchain-core` – core abstractions (prompts, runnables, output parsers)  
  - `langchain-community` – community integrations (FAISS, file loaders)  
  - `langchain-openai` – OpenAI chat models and embeddings  
- **Vector store and text handling**  
  - `faiss-cpu` – local vector database for document embeddings  
  - `langchain-text-splitters` – `MarkdownTextSplitter` for chunking documents  
- **Environment and utilities**  
  - `python-dotenv` – loads API keys from `.env`  
  - `pillow` – used indirectly for image handling if needed

## Code structure

The core logic lives in `scripts/langchain_rag.py` and is organized around a single class:

### `RegulatoryRAG` class

- **Model and embeddings**
  - `self.llm = ChatOpenAI(model="gpt-4o", temperature=0)`  
    GPT‑4o is used as the chat model to generate grounded answers.
  - `self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")`  
    This model turns text chunks into embedding vectors for FAISS.

- **Prompt**
  - A `ChatPromptTemplate` defines the system behavior (“expert in regulatory compliance”) and expects two fields:
    - `{context}` – the retrieved document snippets.  
    - `{input}` – the user’s question.

- **Document ingestion**
  - `load_multiple_documents(file_paths)`  
    Loads `.md` files via `TextLoader`, annotates metadata, splits them with `MarkdownTextSplitter`, and builds a FAISS index saved under `embeddings_cache/vectorstore/`.
  - `load_images_with_vision(markdown_files, image_files)`  
    Optionally analyzes images (e.g. diagrams) with GPT‑4o Vision and turns the descriptions into additional `Document` objects that are indexed alongside the text.

- **Vector store and retriever**
  - `FAISS.from_documents(...)` builds the vector store.  
  - `load_existing_vectorstore()` reloads it from disk on startup to avoid recomputing embeddings every time.  
  - `self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})` exposes a retriever over the indexed chunks.

- **Runnables / retrieval chain**
  - The retrieval chain is built with LangChain Expression Language (LCEL) runnables:
    - `RunnableParallel` queries the retriever and passes through the original question.  
    - A small lambda formats retrieved `Document` objects into a single context string.  
    - `prompt | llm | StrOutputParser()` turns context + question into a plain‑text answer.
  - `query(question: str)` calls this chain and returns the answer string for printing.

### CLI entrypoint

The `main()` function:

- Ensures folders like `embeddings_cache/vectorstore` exist.  
- Tries to load an existing FAISS store; if missing, it scans `documents/` and `images/` to build one.  
- Starts a simple REPL:
  - Prompts: “Your question about the Medical Device Regulation: …”  
  - Sends the question through the RAG pipeline.  
  - Prints a wrapped answer to the terminal.

## Typical usage

1. **Set up environment**
`python3.12 -m venv langchain_env`  
`source langchain_env/bin/activate`

`pip install "langchain>=1.1.0" langchain-core langchain-community langchain-openai`  
`pip install faiss-cpu langchain-text-splitters python-dotenv pillow`  
2. **Prepare project folders**  
`mkdir -p scripts documents images embeddings_cache/vectorstore`

3. **Configuration**

- Place your OpenAI key into a `.env` file in the project root, e.g.:  
`OPENAI_API_KEY=sk-...`

4. **Add content**

- Put your regulatory `.md` files into `documents/`  
- (Optional) Put diagrams into `images/` for vision‑based analysis

5. **Run**
`python scripts/langchain_rag.py`
Then start asking questions about your documents in the terminal

## Extending this demo

Some ideas for further work:

- Add source highlighting (e.g. show which file and section an answer came from)  
- Support multiple regulatory “profiles” (FDA, MDR, IEC 81001‑5‑1) via different document sets  
- Expose the same RAG chain through a simple web UI (FastAPI/Streamlit) instead of the CLI




