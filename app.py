import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import hashlib
from io import BytesIO

def type_text_with_cursor(text: str, delay: float = 0.01):
    """Streamlit typing effect with blinking cursor."""
    container = st.empty()
    typed = ""
    style = """
    <style>
    .typed-text::after {content: "|"; animation: blink 1s infinite;}
    @keyframes blink {0%{opacity:1;}50%{opacity:0;}100%{opacity:1;}}
    .typed-text {white-space: pre-wrap;}
    </style>
    """
    # inject style once
    st.markdown(style, unsafe_allow_html=True)
    for ch in text:
        typed += ch
        container.markdown(f"<div class='typed-text'>{typed}</div>", unsafe_allow_html=True)
        time.sleep(delay)

# --- Load environment variables ---
load_dotenv()

# Get API key from Streamlit secrets or environment
def get_google_api_key():
    """Get Google API key from Streamlit secrets or environment variables."""
    try:
        # Try Streamlit secrets first (for deployment)
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variable (for local development)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("🔑 GOOGLE_API_KEY not found. Please add it to your Streamlit secrets or .env file.")
            st.stop()
        return api_key

def process_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

def get_text_chunks(text):
    """Split text into chunks for processing."""
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks."""
    if not text_chunks:
        st.error("No text chunks to process.")
        return None
    
    try:
        # Set the API key for Google Generative AI
        google_api_key = get_google_api_key()
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        vectorstore = DocArrayInMemorySearch.from_texts(
            texts=text_chunks, 
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    """Create a conversation chain with memory."""
    if not vectorstore:
        return None
    
    try:
        google_api_key = get_google_api_key()
        llm = ChatGoogleGenerativeAI(
            temperature=0.7, 
            model="gemini-2.5-flash",  # Updated model name
            google_api_key=google_api_key,
            convert_system_message_to_human=True
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Use Maximal Marginal Relevance to diversify retrieved chunks
        retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 6}  # Reduced k for better performance
        )

        from langchain.prompts import PromptTemplate
        QA_PROMPT = PromptTemplate(
            template=(
                "You are an expert tutor. Use the following context to answer the user's question. "
                "If the user requests a summary, provide a concise yet thorough summary covering all key points.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
            input_variables=["context", "question"],
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

# ---------------- Utility helpers ----------------
@st.cache_resource(show_spinner="🔄 Indexing document(s)...")
def build_vectorstore(file_bytes_tuple: tuple[bytes, ...]):
    """Return vector store given raw PDF bytes for multiple files (cached)."""
    all_chunks = []
    for b in file_bytes_tuple:
        raw_text = process_pdf(BytesIO(b))
        if raw_text.strip():
            chunks = get_text_chunks(raw_text)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        st.error("No text content found in uploaded PDF(s).")
        return None
    
    return get_vectorstore(all_chunks)

def main():
    st.set_page_config(
        page_title="AI FAQ Chatbot", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI FAQ Chatbot")
    st.markdown("Upload PDF documents and ask questions about their content.")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("📄 Upload Your Documents")
        pdf_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )

        if pdf_files:
            with st.spinner("Processing documents..."):
                file_bytes_list = [f.getvalue() for f in pdf_files]
                file_hash = hashlib.md5(b''.join(file_bytes_list)).hexdigest()

                if st.session_state.get("file_hash") != file_hash:
                    vectorstore = build_vectorstore(tuple(file_bytes_list))
                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.file_hash = file_hash
                        st.success(f"✅ {len(pdf_files)} document(s) processed successfully!")
                    else:
                        st.error("❌ Failed to process documents.")
        
        # Add some helpful information
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("- Upload clear, text-based PDFs")
        st.markdown("- Ask specific questions about the content")
        st.markdown("- Request summaries of sections")

    # Main chat interface
    st.markdown("### 💬 Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if conversation is ready
        if st.session_state.conversation is None:
            with st.chat_message("assistant"):
                st.warning("⚠️ Please upload a PDF document first.")
            return
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.conversation({"question": prompt})
                    answer = response["answer"]
                    
                    # Display the answer with typing effect
                    type_text_with_cursor(answer, delay=0.005)  # Faster typing
                    
                    # Display source documents in an expander
                    if response.get("source_documents"):
                        with st.expander("📚 View Source Documents", expanded=False):
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**📄 Source {i}:**")
                                st.text_area(
                                    f"Content {i}", 
                                    doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                                    height=100,
                                    key=f"source_{i}_{len(st.session_state.messages)}"
                                )
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.info("Please try rephrasing your question or upload a different document.")

if __name__ == "__main__":
    main()