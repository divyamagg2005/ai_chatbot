# AI FAQ Chatbot

An intelligent chatbot that can answer questions based on the content of uploaded PDF documents using LangChain, OpenAI's GPT, and document embeddings.

## Features

- Upload and process PDF documents
- Ask questions in natural language
- Get accurate, context-aware responses
- View source documents for each answer
- Conversation history maintained during the session

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)
3. Upload a PDF document using the sidebar
4. Once the document is processed, start asking questions about its content

## How It Works

1. The application extracts text from the uploaded PDF
2. The text is split into manageable chunks
3. These chunks are converted into vector embeddings using OpenAI's embeddings
4. When you ask a question, the system finds the most relevant chunks using vector similarity search
5. The relevant chunks are passed to GPT-3.5-turbo along with your question to generate a contextual answer

## Note

- The application processes documents locally in your browser
- No data is stored permanently; all processing happens in memory during your session
- You'll need an active internet connection to use the OpenAI API
