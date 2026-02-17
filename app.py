import os
import hashlib
import time
import random
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
from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError

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


# ── Gemini Embeddings with rate-limit handling ────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """
    gemini-embedding-001 with:
      - RETRIEVAL_DOCUMENT task type for indexing
      - RETRIEVAL_QUERY task type for queries
      - Small batches (20 chunks) + sleep to stay within free-tier TPM
      - Exponential backoff retry on 429 / ClientError
    """
    MODEL = "gemini-embedding-001"
    BATCH_SIZE = 10          # 10 chunks × ~200 tokens = ~2,000 tokens/batch
    BATCH_SLEEP = 4.0        # 4s sleep → stays well under 30,000 TPM free limit
    MAX_RETRIES = 6

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def _embed_with_retry(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Single batch embed with exponential backoff on rate-limit errors."""
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self._client.models.embed_content(
                    model=self.MODEL,
                    contents=texts,
                    config=genai_types.EmbedContentConfig(task_type=task_type),
                )
                return [e.values for e in result.embeddings]
            except ClientError as e:
                # 429 = quota exceeded, 503 = overloaded
                if "429" in str(e) or "503" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    # Wait long enough for the TPM window to reset (min 15s)
                    wait = max(15.0, (2 ** attempt)) + random.uniform(0, 2)
                    st.toast(f"⏳ Rate limit hit — waiting {wait:.0f}s for quota reset…", icon="⚠️")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Embedding failed after {self.MAX_RETRIES} retries.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_emb = []
        total_batches = (len(texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        for i, start in enumerate(range(0, len(texts), self.BATCH_SIZE)):
            batch = texts[start : start + self.BATCH_SIZE]
            all_emb.extend(self._embed_with_retry(batch, "RETRIEVAL_DOCUMENT"))
            # Sleep between batches to avoid bursting the free-tier TPM
            if i < total_batches - 1:
                time.sleep(self.BATCH_SLEEP)
        return all_emb

    def embed_query(self, text: str) -> list[float]:
        return self._embed_with_retry([text], "RETRIEVAL_QUERY")[0]


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
        chunk_size=800, chunk_overlap=100
    ).split_text(text)


# ── Build retriever (cached by file hash) ─────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_retriever(file_bytes_tuple: tuple[bytes, ...]):
    api_key = get_google_api_key()

    all_chunks: list[str] = []
    for raw in file_bytes_tuple:
        all_chunks.extend(split_text(extract_text_from_pdf(BytesIO(raw))))

    if not all_chunks:
        st.error("No readable text found in the uploaded PDF(s).")
        return None

    total_batches = (len(all_chunks) + GeminiEmbeddings.BATCH_SIZE - 1) // GeminiEmbeddings.BATCH_SIZE
    est_seconds = total_batches * GeminiEmbeddings.BATCH_SLEEP
    st.info(
        f"📄 Found **{len(all_chunks)} chunks** across your PDFs. "
        f"Indexing with rate-limit pacing (~{est_seconds:.0f}s)…"
    )

    progress = st.progress(0, text="Embedding chunks…")

    class ProgressEmbeddings(GeminiEmbeddings):
        def embed_documents(self, texts):
            all_emb = []
            total = (len(texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
            for i, start in enumerate(range(0, len(texts), self.BATCH_SIZE)):
                batch = texts[start : start + self.BATCH_SIZE]
                all_emb.extend(self._embed_with_retry(batch, "RETRIEVAL_DOCUMENT"))
                pct = int((i + 1) / total * 100)
                progress.progress(pct, text=f"Embedding… batch {i+1}/{total}")
                if i < total - 1:
                    time.sleep(self.BATCH_SLEEP)
            progress.empty()
            return all_emb

    vectorstore = FAISS.from_texts(all_chunks, embedding=ProgressEmbeddings(api_key))

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


# ── LCEL chain ────────────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def run_chain(retriever, llm, question: str, chat_history: list) -> tuple[str, list]:
    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the context below. "
            "If the answer is not in the context, say so honestly.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
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
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("file_hash", None)

    api_key = get_google_api_key()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("📄 Upload Documents")
        pdf_files = st.file_uploader(
            "Choose PDF file(s)", type="pdf",
            accept_multiple_files=True,
            help="Text-based PDFs work best. Large files take longer to index on free tier.",
        )

        if pdf_files:
            file_bytes = [f.getvalue() for f in pdf_files]
            file_hash = hashlib.md5(b"".join(file_bytes)).hexdigest()

            if st.session_state.file_hash != file_hash:
                try:
                    retriever = build_retriever(tuple(file_bytes))
                    if retriever:
                        st.session_state.retriever = retriever
                        st.session_state.file_hash = file_hash
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.success(f"✅ {len(pdf_files)} document(s) ready!")
                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")

        st.markdown("---")
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("### 💡 Tips")
        st.markdown("- Large PDFs take ~1–2 min to index on free tier")
        st.markdown("- Use text-based (not scanned) PDFs")
        st.markdown("- Ask specific questions for precise answers")

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

                st.write_stream(word_stream(answer))

                # Update history, cap at 10 turns
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=answer),
                ])
                st.session_state.chat_history = st.session_state.chat_history[-20:]

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
