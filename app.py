import streamlit as st
import os
GOOGLE_API_KEY = "AIzaSyBHTV8_2Ul2nrKdLEht5BKWbQEkgIZvqIA"
from dotenv import load_dotenv
from pypdf import PdfReader # Using PdfReader as PyPDFLoader can sometimes be slow or have issues
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.embeddings import FastEmbeddings
from langchain.vectorstores import FAISS # In-memory vector store
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
import time
import html
import random
from typing import Generator

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in the .env file or Streamlit secrets.")
    st.stop()

# --- Helper Functions ---

@st.cache_data(show_spinner="Extracting text from PDFs...") # Cache PDF text extraction
def get_pdf_text(pdf_docs):
    """Extracts text content from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text: # Add text only if extraction was successful
                    text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

@st.cache_data(show_spinner="Splitting documents into chunks...") # Cache text chunking
def get_text_chunks(text):
    """Splits text into manageable chunks."""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Adjust chunk size as needed
        chunk_overlap=200, # Keep some overlap for context continuity
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Use st.cache_resource for non-data objects like models and vector stores
def get_vectorstore(_text_chunks):
    """Creates an in-memory FAISS vector store from text chunks."""
    if not _text_chunks:
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY) # Efficient embedding model
        vectorstore = FAISS.from_texts(texts=_text_chunks, embedding=embeddings)
        print("vectorstore created")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        # Provide more specific guidance if it's an API key issue
        if "API key not valid" in str(e):
             st.error("Please ensure your Google API Key is correct and has the Generative Language API enabled.")
        return None

def get_retrieval_chain(vectorstore):
    """Creates the Langchain retrieval chain."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True,
            stream=True  # Enable streaming
        )

        # Updated prompt for concise answers
        prompt_template = """
You are an intelligent AI assistant focused on providing helpful and accurate information.

Role: Respond as a knowledgeable guide who:
- Analyzes provided context thoroughly
- Draws relevant connections
- Explains concepts clearly
- Maintains objectivity
- Cites information from the source material when possible

Guidelines:
1. Start with a direct answer to the question
2. Support key points with relevant context
3. Use clear, accessible language
4. Structure responses for easy reading
5. Stay focused on the topic at hand
6. Acknowledge limitations when appropriate

Context: {context}

Question: {input}

Please provide a clear and informative response that helps the user understand the topic:
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create the document combining chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Create the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks

        # Create the main retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain

    except Exception as e:
        st.error(f"Error creating retrieval chain: {e}")
        return None

def generate_response(response: str) -> Generator[str, None, None]:
    """Generate response chunks for streaming."""
    words = response.split()
    for i, word in enumerate(words):
        yield word + " "
        if i % 10 == 0:  # Add natural pauses every 10 words
            yield ""

# --- Streamlit App ---
st.set_page_config(
    page_title="EduBot - Your Educational Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "current_subject" not in st.session_state:
    st.session_state.current_subject = "General"

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .message-timestamp {
        font-size: 0.8rem;
        color: #666;
    }
    .message-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .quick-prompts {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-wrapper {
        max-width: 800px;
        margin: 0 auto;
    }
    .stChatMessage {
        background-color: transparent !important;
        padding: 1rem 0 !important;
    }
    .user-message-wrapper {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    .assistant-message-wrapper {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1rem;
    }
    .message-content {
        padding: 0.8rem 1.2rem;
        border-radius: 15px;
        max-width: 80%;
        line-height: 1.4;
    }
    .user-message {
        background-color: #DCF8C6;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .assistant-message {
        background-color: #FFFFFF;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .message-timestamp {
        font-size: 0.7rem;
        color: #999;
        margin-top: 0.2rem;
        text-align: right;
    }
    .chat-container {
        background-color: #E5DDD5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        height: 600px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }
    .message-bubble {
        max-width: 80%;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 1rem;
        border-radius: 15px;
        position: relative;
        animation: fadeIn 0.3s ease-in;
    }
    .user-bubble {
        background-color: #DCF8C6;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .assistant-bubble {
        background-color: #FFFFFF;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .message-text {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    </style>
    <script>
        function scrollToBottom() {
            const chatContainer = document.querySelector('.chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
    </script>
""", unsafe_allow_html=True)

# Header Section
col1, col2 = st.columns([2, 3])
with col1:
    st.image("https://raw.githubusercontent.com/yourusername/EDubot/main/assets/edubot-logo.png", width=100)
with col2:
    st.title("🎓 EduBot - Your Educational Assistant")
    st.markdown("Transform your learning experience with AI-powered document interactions!")

# --- Sidebar Enhancement ---
with st.sidebar:
    st.header("📚 Learning Resources")
    
    # Subject Selection
    # subject = st.selectbox(
    #     "Select Subject Area",
    #     ["General", "Mathematics", "Science", "Literature", "History", "Computer Science"]
    # )
    
    # File Upload Section
    st.subheader("📄 Upload Study Materials")
    pdf_docs = st.file_uploader(
        "Upload PDF files (Notes, Textbooks, etc.)",
        accept_multiple_files=True,
        type="pdf"
    )
    
    # Processing Options
    with st.expander("⚙️ Processing Options"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        overlap_size = st.slider("Overlap Size", 50, 500, 200)
    
    process_button = st.button("🔄 Process Documents")

    if process_button:
        if pdf_docs:
            # 1. Extract Text
            raw_text = get_pdf_text(pdf_docs)

            if raw_text:
                # 2. Split into Chunks
                text_chunks = get_text_chunks(raw_text)

                if text_chunks:
                    # 3. Create Vector Store (triggers embedding)
                    vectorstore = get_vectorstore(text_chunks) # Pass chunks to trigger caching correctly

                    if vectorstore:
                        st.success(f"Processed {len(pdf_docs)} PDF(s) successfully!")
                        # 4. Create Retrieval Chain and store in session state
                        st.session_state.retrieval_chain = get_retrieval_chain(vectorstore)
                        st.session_state.processed_files = [pdf.name for pdf in pdf_docs] # Store names
                        st.session_state.chat_history = [] # Reset chat history on new processing
                        st.rerun() # Rerun to update main page state
                    else:
                        st.error("Failed to create vector store. Please check API key and logs.")
                else:
                    st.warning("Could not extract any text chunks from the PDFs.")
            else:
                st.warning("Could not extract text from the uploaded PDFs.")
        else:
            st.warning("Please upload at least one PDF file.")

    # Display processed files
    if "processed_files" in st.session_state:
        st.subheader("📑 Processed Materials:")
        for file_name in st.session_state.processed_files:
            st.markdown(f"📘 {file_name}")

# --- Main Chat Interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Study Session Info
if st.session_state.retrieval_chain is not None:
    st.markdown("### 📝 Current Study Session")
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.info(f"Subject: General")
    # with col2:
    st.info(f"Documents Loaded: {len(st.session_state.processed_files) if 'processed_files' in st.session_state else 0}")

# Updated Chat Container
st.markdown("### 💬 Chat Interface")
chat_placeholder = st.empty()

MAX_HISTORY = 15  # Maximum number of Q&A pairs to show

def display_chat_messages():
    # Get last 30 messages (5 Q&A pairs)
    recent_messages = st.session_state.chat_history[-MAX_HISTORY*2:] if st.session_state.chat_history else []
    
    for message in recent_messages:
        is_user = isinstance(message, HumanMessage)
        with st.chat_message("user" if is_user else "assistant", 
                           avatar="👤" if is_user else "🎓"):
            st.markdown(message.content)

# Display chat messages
display_chat_messages()

# Enhanced Input Area
if st.session_state.retrieval_chain is not None:
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        # Add user message
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        
        try:
            with st.chat_message("assistant"):
                # Get response
                response = st.session_state.retrieval_chain.invoke(
                    {"input": user_question},
                    config={"callbacks": []}
                )
                answer = response.get("answer", "")
                
                # Stream the response
                st.write_stream(generate_response(answer))
                
                # Add to chat history
                st.session_state.chat_history.append(AIMessage(content=answer))
                
                # Limit chat history
                if len(st.session_state.chat_history) > MAX_HISTORY * 2:
                    st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY*2:]
                
                # Rerun to update display
                st.rerun()
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.info("👆 Start by uploading your study materials in the sidebar!")

# Enhanced Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center'>
#         <h4>🎓 EduBot - Your AI Study Companion</h4>
#         <p>Making learning interactive, engaging, and personalized</p>
#         <div style='font-size: small; color: #666; margin-top: 1rem;'>
#             Powered by Gemini AI 🤖 | Built for student success 📚
#         </div>
#     </div>
# """, unsafe_allow_html=True)