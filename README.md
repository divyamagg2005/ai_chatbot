# 🤖 AI FAQ Chatbot

An intelligent chatbot that can answer questions based on the content of uploaded PDF documents using Google's Gemini model, LangChain, and document embeddings. This application provides a user-friendly interface for interacting with your documents through natural language queries.

## ✨ Features

- 📄 Upload and process multiple PDF documents
- 💬 Natural language question answering
- 🧠 Context-aware responses with conversation memory
- 🔍 View source documents for each answer
- ⚡ Fast and efficient document processing
- 🌟 Modern, responsive UI with typing animation
- 🔒 Secure API key management

## 🚀 Prerequisites

- Python 3.8 or higher
- Google API key with access to Gemini models
- [Google AI Studio](https://makersuite.google.com/app/apikey) account for API key

## 🛠️ Setup

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/ai-faq-chatbot.git
   cd ai-faq-chatbot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment:
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```
   - Alternatively, you can store the key in `/.streamlit/secrets.toml` (recommended for deployment):
  ```toml
  [default]
  GOOGLE_API_KEY="your_google_api_key_here"
  ```

## 🏃‍♂️ Running the Application

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to upload one or more PDF documents

4. Once processing is complete, start asking questions about the document content

## 🧠 How It Works

1. **Document Processing**
   - PDFs are parsed and text is extracted
   - Text is split into manageable chunks with overlap for context
   - Chunks are converted to vector embeddings using Google's embedding model

2. **Question Answering**
   - Your question is converted to an embedding
   - The system performs a similarity search to find relevant document chunks
   - The most relevant context is passed to Google's Gemini model
   - The model generates a natural language response based on the context

3. **Conversation Flow**
   - The application maintains conversation history for context
   - Each response includes source document references
   - The UI provides visual feedback during processing

## 🔧 Technical Stack

- **Framework**: Streamlit
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Google's embedding-001
- **Vector Store**: DocArrayInMemorySearch
- **Document Processing**: PyPDF2, LangChain
- **Environment Management**: python-dotenv

## 📝 Notes

- The application processes documents locally in your browser
- No document data is stored on any server
- For best results, use well-formatted PDFs with clear text (not scanned images)
- The application supports multiple document uploads for cross-document queries

## 🚢 Deployment

The project can be deployed to any platform that supports Streamlit (e.g. Streamlit Community Cloud, Hugging Face Spaces, or Docker). A `.devcontainer` configuration is provided for a one-click launch in GitHub Codespaces.

```bash
# Example – Streamlit Community Cloud
1. Fork or push this repository to your GitHub account.
2. Go to https://share.streamlit.io and create a new app pointing to `app.py`.
3. Add `GOOGLE_API_KEY` to the app secrets.
4. Click “Deploy”.
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- No data is stored permanently; all processing happens in memory during your session
- You'll need an active internet connection to call the Google Generative AI API


This is the deployed link. you can go to the link and test the application. https://aipdfchatbotsystem.streamlit.app/
