import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Dynamic RAG Chatbot", page_icon="🤖")
st.title("Dynamic RAG Chatbot with User-Uploaded PDF")
st.write("Upload a PDF, then ask questions based on its content!")

# --- Configuration and Initialization ---
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

TEMP_QDRANT_COLLECTION_NAME = "user_uploaded_pdf_vectors"

@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash")

embedding_model = get_embedding_model()
llm = get_llm()

# --- PDF Upload and Processing Logic ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and st.session_state.vector_db is None:
    st.info("Processing PDF... This might take a moment.")
    
    temp_dir = "temp_uploaded_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        loader = PyPDFLoader(file_path=temp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents=docs)

        with st.spinner("Creating vector database from PDF content..."):
            st.session_state.vector_db = QdrantVectorStore.from_documents(
                documents=split_docs,
                url=QDRANT_URL,
                collection_name=TEMP_QDRANT_COLLECTION_NAME,
                embedding=embedding_model,
                force_recreate=True,
                api_key=QDRANT_API_KEY
            )
        st.success("PDF processed and vector database ready! You can now ask questions.")

        st.session_state.messages = []
        st.session_state.messages.append(SystemMessage(content="""
            You are a helpful AI Assistant who answers user queries based on the available context
            retrieved from a PDF file along with page_contents and page number.

            You should only answer the user based on the following context and navigate the user
            to open the right page number to know more.
        """))
        st.session_state.messages.append(AIMessage(content="I have processed the PDF. How can I help you with its content?"))

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

# --- Chat Interface ---
if st.session_state.vector_db is not None:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if prompt := st.chat_input("What would you like to know about the PDF?"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                search_results = st.session_state.vector_db.similarity_search(query=prompt)

                current_turn_context = "\n\n\n".join([
                    f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
                    for result in search_results
                ])

                current_system_message = SystemMessage(
                    content=st.session_state.messages[0].content + "\n\nContext:\n" + current_turn_context
                )

                messages_to_send = [current_system_message] + st.session_state.messages[1:] + [HumanMessage(content=prompt)]

                chat_completion = llm.invoke(messages_to_send)
                ai_response_content = chat_completion.content
                st.markdown(ai_response_content)
                st.session_state.messages.append(AIMessage(content=ai_response_content))