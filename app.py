from __future__ import annotations

import io
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBED_MODELS = {
    "nvidia/nemotron-3-embed-1b:free": "Free · text retrieval / RAG",
}


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
def get_llm(
    api_key: str,
    model: str,
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Created a chat model using OpenRouter's OpenAI-compatible API."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        streaming=True,
        default_headers={"X-Title": "LangChain RAG Chatbot"},
    )


def get_embeddings(api_key: str, model: str) -> OpenAIEmbeddings:
    """Created an embeddings client using OpenRouter."""
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        # Send text directly instead of converting it to token IDs locally.
        check_embedding_ctx_length=False,
        # Request ordinary floating-point vectors.
        model_kwargs={"encoding_format": "float"},
        chunk_size=64,
        max_retries=6,
    )


# ---------------------------------------------------------------------
# Document ingestion: load -> split -> embed -> store
# ---------------------------------------------------------------------
def load_uploaded_file(name: str, data: bytes) -> list[Document]:
    """Converted an uploaded PDF, TXT, or Markdown file into Documents."""
    extension = os.path.splitext(name)[1].lower()

    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        documents = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": name,
                            "page": page_number,  # zero-based internally
                        },
                    )
                )

        return documents

    if extension in {".txt", ".md"}:
        text = data.decode("utf-8", errors="ignore")
        if not text.strip():
            return []
        return [Document(page_content=text, metadata={"source": name})]

    raise ValueError(f"Unsupported file type: {extension}")


def build_vectorstore(
    documents: list[Document],
    embeddings: OpenAIEmbeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> tuple[InMemoryVectorStore, int]:
    """Splitted documents, embed the chunks, and store them in memory."""
    if not documents:
        raise ValueError("No readable text was found in the uploaded files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("The uploaded documents did not produce any text chunks.")

    vectorstore = InMemoryVectorStore(embedding=embeddings)
    vectorstore.add_documents(chunks)

    return vectorstore, len(chunks)


# ---------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation and the user's latest question, rewrite "
            "the latest question as one standalone search query. "
            "Return only the query.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about the "
            "user's documents.\n"
            "Use ONLY the context below. If the answer is not in the context, "
            "say that you do not know based on the uploaded documents.\n"
            "Cite sources inline using [source: filename, p.N] when page "
            "numbers are available.\n\n"
            "Context:\n{context_text}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def format_docs(documents: list[Document]) -> str:
    """Formatted retrieved chunks and included their source metadata."""
    formatted_parts = []

    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")

        if isinstance(page, int):
            source_label = f"{source}, p.{page + 1}"
        else:
            source_label = source

        formatted_parts.append(
            f"[source: {source_label}]\n{document.page_content}"
        )

    return "\n\n---\n\n".join(formatted_parts)


def build_chain(
    llm: ChatOpenAI,
    vectorstore: InMemoryVectorStore,
    k: int = 4,
):
    """Built a chain that retrieves context and streams the generated answer."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

    def make_query(inputs: dict[str, Any]) -> str:
        # Rewrite follow-up questions only when there is chat history.
        if inputs.get("chat_history"):
            return rewrite_chain.invoke(inputs).strip()
        return inputs["question"]

    answer_chain = (
        RunnablePassthrough.assign(
            context_text=lambda inputs: format_docs(inputs["context"])
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(
            query=RunnableLambda(make_query)
        )
        .assign(
            context=lambda inputs: retriever.invoke(inputs["query"])
        )
        .assign(
            answer=answer_chain
        )
    )


def to_lc_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    """Converted Streamlit chat history into LangChain messages."""
    converted_messages: list[BaseMessage] = []

    for message in history:
        if message["role"] == "user":
            converted_messages.append(HumanMessage(content=message["content"]))
        else:
            converted_messages.append(AIMessage(content=message["content"]))

    return converted_messages


# ---------------------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Chatbot",
    page_icon="📚",
    layout="wide",
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("indexed_files", [])
st.session_state.setdefault("index_embed_model", None)

# Sidebar: settings and document upload
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenRouter API key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
    )

    configured_chat_model = os.getenv("CHAT_MODEL", "openrouter/free").strip()
    if configured_chat_model == "nvidia/nemotron-3-ultra-550b-a55b:free":
        configured_chat_model = "openrouter/free"

    chat_model = st.text_input(
        "Chat model",
        value=configured_chat_model,
        help="Use openrouter/free to let OpenRouter select an available free chat model.",
    )

    embedding_options = list(EMBED_MODELS)
    env_embedding_model = os.getenv("EMBED_MODEL")

    if env_embedding_model and env_embedding_model not in embedding_options:
        embedding_options.insert(0, env_embedding_model)

    default_embedding_index = (
        embedding_options.index(env_embedding_model)
        if env_embedding_model in embedding_options
        else 0
    )

    embedding_model = st.selectbox(
        "Embedding model",
        embedding_options,
        index=default_embedding_index,
        format_func=lambda model_id: (
            f"{model_id} ({EMBED_MODELS.get(model_id, 'custom model')})"
        ),
        help="Changing this requires rebuilding the document index.",
    )

    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    st.divider()
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button(
        "Build index",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        if not api_key:
            st.error("Add your OpenRouter API key first.")
        else:
            try:
                with st.spinner("Reading, splitting, and embedding documents..."):
                    all_documents: list[Document] = []

                    for uploaded_file in uploaded_files:
                        all_documents.extend(
                            load_uploaded_file(
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                            )
                        )

                    embeddings = get_embeddings(api_key, embedding_model)
                    vectorstore, number_of_chunks = build_vectorstore(
                        all_documents,
                        embeddings,
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.indexed_files = [
                        uploaded_file.name for uploaded_file in uploaded_files
                    ]
                    st.session_state.index_embed_model = embedding_model

                st.success(
                    f"Indexed {len(uploaded_files)} file(s) "
                    f"into {number_of_chunks} chunks."
                )

            except Exception as error:
                st.error(f"Indexing failed: {error}")

    if st.session_state.indexed_files:
        st.caption(
            "Indexed: "
            + ", ".join(st.session_state.indexed_files)
            + f"\n\nEmbedding model: {st.session_state.index_embed_model}"
        )

        if st.session_state.index_embed_model != embedding_model:
            st.warning(
                "The embedding model has changed. Click **Build index** "
                "to rebuild the index with the selected model."
            )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat area
st.title("Chatbot")
st.caption("LangChain · OpenRouter · Streamlit")


def show_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source['label']}**")
            source_text = source["text"]
            st.text(
                source_text[:500] + ("..." if len(source_text) > 500 else "")
            )


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))


is_ready = bool(api_key and st.session_state.vectorstore)
chat_placeholder = (
    "Ask about your documents..."
    if is_ready
    else "Add an API key and build an index first."
)

question = st.chat_input(
    chat_placeholder,
    disabled=not is_ready,
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    chat_history = to_lc_messages(st.session_state.messages[:-1])

    try:
        chain = build_chain(
            get_llm(api_key, chat_model),
            st.session_state.vectorstore,
            k=top_k,
        )

        chain_inputs = {
            "question": question,
            "chat_history": chat_history,
        }

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer = ""
            retrieved_context: list[Document] = []

            # The chain emits retrieved context and then answer tokens.
            for chunk in chain.stream(chain_inputs):
                if "context" in chunk:
                    retrieved_context = chunk["context"]

                if "answer" in chunk:
                    answer += chunk["answer"]
                    answer_placeholder.markdown(answer + "▌")

            answer_placeholder.markdown(answer or "No answer was generated.")

            source_items = []
            for document in retrieved_context:
                page = document.metadata.get("page")
                label = document.metadata.get("source", "unknown")

                if isinstance(page, int):
                    label += f" — page {page + 1}"

                source_items.append(
                    {
                        "label": label,
                        "text": document.page_content,
                    }
                )

            show_sources(source_items)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_items,
            }
        )

    except Exception as error:
        error_message = f"⚠️ Error: {error}"

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "sources": [],
            }
        )
