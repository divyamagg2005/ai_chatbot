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
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
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

# ── Custom Embeddings class with task_type support ────────────────────────────
# LangChain's built-in wrapper doesn't distinguish query vs document task types.
# Using task types significantly improves RAG retrieval accuracy per Google docs.

class GeminiEmbeddings(Embeddings):
    """
    Wraps gemini-embedding-001 with correct task types:
      - RETRIEVAL_DOCUMENT  →  used when indexing PDF chunks
      - RETRIEVAL_QUERY     →  used when embedding the user's question
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

    # Called when INDEXING documents
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch in groups of 100 (safe limit per call)
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed(batch, "RETRIEVAL_DOCUMENT"))
        return all_embeddings

    # Called when QUERYING
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


# ── Pipeline (cached by file content hash) ───────────────────────────────────

@st.cache_resource(show_spinner="🔄 Indexing documents…")
def build_chain(file_bytes_tuple: tuple[bytes, ...]):
    api_key = get_google_api_key()

    # 1. Extract + chunk text
    all_chunks: list[str] = []
    for raw in file_bytes_tuple:
        text = extract_text_from_pdf(BytesIO(raw))
        all_chunks.extend(split_text(text))

    if not all_chunks:
        st.error("No readable text found in the uploaded PDF(s).")
        return None

    # 2. Build FAISS index — uses RETRIEVAL_DOCUMENT task type for indexing
    embeddings = GeminiEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_texts(all_chunks, embedding=embeddings)

    # 3. LLM — gemini-2.5-flash is the current stable balanced model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )

    # 4. Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # 5. MMR retriever for diversity across retrieved chunks
    #    Uses RETRIEVAL_QUERY task type automatically via embed_query()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15},
    )

    # 6. Prompt
    qa_prompt = PromptTemplate(
        template=(
            "You are a helpful and knowledgeable assistant. "
            "Use the context below to answer the question accurately and concisely. "
            "If the answer isn't in the context, say so honestly.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return chain


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

    st.session_state.setdefault("chain", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("file_hash", None)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("📄 Upload Documents")
        pdf_files = st.file_uploader(
            "Choose PDF file(s)",
            type="pdf",
            accept_multiple_files=True,
            help="Text-based PDFs work best.",
        )

        if pdf_files:
            file_bytes = [f.getvalue() for f in pdf_files]
            file_hash = hashlib.md5(b"".join(file_bytes)).hexdigest()

            if st.session_state.file_hash != file_hash:
                chain = build_chain(tuple(file_bytes))
                if chain:
                    st.session_state.chain = chain
                    st.session_state.file_hash = file_hash
                    st.session_state.messages = []
                    st.success(f"✅ {len(pdf_files)} document(s) ready!")
                else:
                    st.error("❌ Failed to process documents.")

        st.markdown("---")
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            if st.session_state.chain:
                st.session_state.chain.memory.clear()
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

        if st.session_state.chain is None:
            with st.chat_message("assistant"):
                st.warning("⚠️ Please upload a PDF first.")
            return

        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Thinking…"):
                    result = st.session_state.chain({"question": prompt})

                answer: str = result["answer"]
                st.write_stream(word_stream(answer))

                source_docs = result.get("source_documents", [])
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
