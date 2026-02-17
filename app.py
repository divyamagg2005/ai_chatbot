import os
import hashlib
import time
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from google import genai
from google.genai import types as genai_types

load_dotenv(override=True)

# ── API key ───────────────────────────────────────────────────────────────────

def get_google_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "").strip()
    except Exception:
        pass
    if not key:
        key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        st.error("🔑 GOOGLE_API_KEY not found. Add it to Streamlit secrets or a .env file.")
        st.stop()
    os.environ["GOOGLE_API_KEY"] = key
    return key


# ── Gemini Embeddings with task_type support ──────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """
    Uses gemini-embedding-001 with correct task types:
      RETRIEVAL_DOCUMENT → indexing PDF chunks
      RETRIEVAL_QUERY    → embedding the user's question
    """
    MODEL = "gemini-embedding-001"

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=texts,
            config=genai_types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in result.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_emb = []
        for i in range(0, len(texts), 100):
            all_emb.extend(self._embed(texts[i : i + 100], "RETRIEVAL_DOCUMENT"))
        return all_emb

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


# ── PDF helpers ───────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_like) -> str:
    try:
        reader = PdfReader(file_like)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def split_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_text(text)


# ── Build retriever (cached by file hash) ─────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Indexing documents…")
def build_retriever(file_bytes_tuple: tuple[bytes, ...]):
    api_key = get_google_api_key()

    all_chunks: list[str] = []
    for raw in file_bytes_tuple:
        all_chunks.extend(split_text(extract_text_from_pdf(BytesIO(raw))))

    if not all_chunks:
        st.error("No readable text found in the uploaded PDF(s).")
        return None

    vectorstore = FAISS.from_texts(all_chunks, embedding=GeminiEmbeddings(api_key))
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15},
    )


def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )


# ── LCEL chain helpers ────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the context below. "
            "If the answer is not in the context, say so honestly.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


def run_chain(retriever, llm, question: str, chat_history: list) -> tuple[str, list]:
    """Run one turn of RAG + return (answer, source_docs)."""
    # Retrieve
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build and invoke chain
    chain = (
        build_prompt()
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })

    return answer, docs


# ── Streaming helper ──────────────────────────────────────────────────────────

def word_stream(text: str, delay: float = 0.018):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="PDF Chat", page_icon="🤖", layout="wide")
    st.title("🤖 AI PDF Chatbot")
    st.caption("Upload one or more PDFs and chat with their content.")

    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("messages", [])        # {"role", "content"}
    st.session_state.setdefault("chat_history", [])    # LangChain message objects
    st.session_state.setdefault("file_hash", None)

    api_key = get_google_api_key()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("📄 Upload Documents")
        pdf_files = st.file_uploader(
            "Choose PDF file(s)", type="pdf",
            accept_multiple_files=True,
            help="Text-based PDFs work best.",
        )

        if pdf_files:
            file_bytes = [f.getvalue() for f in pdf_files]
            file_hash = hashlib.md5(b"".join(file_bytes)).hexdigest()

            if st.session_state.file_hash != file_hash:
                retriever = build_retriever(tuple(file_bytes))
                if retriever:
                    st.session_state.retriever = retriever
                    st.session_state.file_hash = file_hash
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.success(f"✅ {len(pdf_files)} document(s) ready!")
                else:
                    st.error("❌ Failed to process documents.")

        st.markdown("---")
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("### 💡 Tips")
        st.markdown("- Use text-based (not scanned) PDFs for best results")
        st.markdown("- Ask specific questions for more precise answers")
        st.markdown("- Ask for *summaries* or *key points* for an overview")

    # ── Chat area ─────────────────────────────────────────────────────────────
    st.markdown("### 💬 Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your document(s)…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.retriever is None:
            with st.chat_message("assistant"):
                st.warning("⚠️ Please upload a PDF first.")
            return

        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Thinking…"):
                    llm = get_llm(api_key)
                    answer, source_docs = run_chain(
                        st.session_state.retriever,
                        llm,
                        prompt,
                        st.session_state.chat_history,
                    )

                # Stream the answer
                st.write_stream(word_stream(answer))

                # Update LangChain message history
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=answer),
                ])
                # Keep history bounded (last 10 turns = 20 messages)
                st.session_state.chat_history = st.session_state.chat_history[-20:]

                # Source chunks
                if source_docs:
                    with st.expander("📚 Source Chunks", expanded=False):
                        for i, doc in enumerate(source_docs, 1):
                            preview = doc.page_content[:600]
                            if len(doc.page_content) > 600:
                                preview += "…"
                            st.markdown(f"**Chunk {i}**")
                            st.text(preview)
                            if i < len(source_docs):
                                st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Try rephrasing your question or re-uploading the document.")


if __name__ == "__main__":
    main()
