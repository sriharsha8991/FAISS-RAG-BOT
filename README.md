# EduBot - AI-Powered Educational Assistant

An intelligent document-based chat application that helps students learn from their study materials using Gemini AI and LangChain.

## 🏗️ Architecture Overview

### 1. Technology Stack
- **Frontend**: Streamlit
- **Backend Processing**: LangChain
- **AI Model**: Google Gemini AI
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Text Processing**: PyPDF2, LangChain Text Splitters
- **Embeddings**: Google Generative AI Embeddings

### 2. Core Components

#### 2.1 Document Processing Pipeline
```
PDF Documents → Text Extraction → Chunk Splitting → Embeddings → Vector Store
```

1. **Text Extraction** (`get_pdf_text`):
   - Uses PyPDF2 for PDF parsing
   - Handles multiple documents
   - Caches results for performance

2. **Text Chunking** (`get_text_chunks`):
   - Implements RecursiveCharacterTextSplitter
   - Chunk size: 1000 characters
   - Overlap: 200 characters
   - Ensures context continuity

3. **Vector Store Creation** (`get_vectorstore`):
   - Uses Google AI embeddings
   - Creates FAISS index
   - In-memory storage
   - Cached using Streamlit's cache_resource

#### 2.2 Chat System Architecture

```
User Query → Retrieval Chain → Context Matching → LLM Processing → Streamed Response
```

1. **Retrieval Chain** (`get_retrieval_chain`):
   - Combines document retrieval and LLM
   - Uses Gemini Flash model
   - Implements streaming responses
   - Custom educational prompt template

2. **Chat Interface**:
   - Limited to 5 Q&A pairs
   - Real-time streaming responses
   - Message persistence using session state
   - Automatic scrolling

### 3. State Management

- **Session State Variables**:
  ```python
  retrieval_chain: LangChain Retrieval Chain
  chat_history: List[Messages]
  processed_files: List[str]
  current_subject: str
  ```

### 4. User Interface Components

1. **Sidebar**:
   - File upload interface
   - Processing controls
   - Document list

2. **Main Interface**:
   - Chat container
   - Message display
   - Input area
   - Status information

### 5. Key Features

1. **Document Processing**:
   - Multi-document support
   - Automatic text chunking
   - Efficient vector storage

2. **Chat Functionality**:
   - Context-aware responses
   - Stream-based output
   - History management
   - Message formatting

3. **User Experience**:
   - Real-time feedback
   - Error handling
   - Progress indicators
   - Responsive design

## 🚀 Implementation Details

### 1. Document Processing

```python
PDF Upload → get_pdf_text() → get_text_chunks() → get_vectorstore()
```

### 2. Query Processing

```python
User Input → Retrieval Chain → Document Search → Context Merge → LLM Generation → Streamed Output
```

### 3. Response Generation

1. Query vectorization
2. Similarity search in FAISS
3. Context retrieval
4. LLM processing
5. Streamed response

## 💻 Technical Specifications

1. **Dependencies**:
   - streamlit==1.31.0
   - langchain & components
   - google-generativeai
   - faiss-cpu
   - pypdf

2. **Model Configuration**:
   - Model: gemini-2.0-flash
   - Temperature: 0.3
   - Streaming enabled
   - Context window: 5 chunks

3. **Performance Optimizations**:
   - Caching for PDF processing
   - Resource caching for vector store
   - Efficient chunk management
   - Limited chat history

## 🔧 Setup and Configuration

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **API Configuration**:
   - Create .env file
   - Add GOOGLE_API_KEY

3. **Running the App**:
   ```bash
   streamlit run app.py
   ```

## 🔒 Security Considerations

1. API key management using .env
2. Input validation
3. Error handling
4. Rate limiting
5. Session state management

## 🎯 Best Practices

1. Document chunking strategies
2. Prompt engineering
3. Error handling
4. User feedback
5. Performance optimization
