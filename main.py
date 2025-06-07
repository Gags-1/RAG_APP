from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Import message classes, including AIMessage

load_dotenv()

# --- Qdrant and Embedding Setup ---
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

# Initialize the Gemini Chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

print("AI Chatbot Ready! Type 'exit' or 'quit' to end the conversation.")

# --- Initialize chat history ---
chat_history = []


initial_system_prompt = """
    You are a helpful AI Assistant who answers user queries based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the user
    to open the right page number to know more.
"""
chat_history.append(SystemMessage(content=initial_system_prompt))


# --- Continuous Chat Loop ---
while True:
    query = input("> ")

    if query.lower() in ["exit", "quit"]:
        print("🤖: Goodbye!")
        break

  
    search_results = vector_db.similarity_search(
        query=query
    )


    current_turn_context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

    messages_for_current_turn = [
        SystemMessage(content=initial_system_prompt + "\n\nContext:\n" + current_turn_context)
    ]

  
    messages_for_current_turn.extend(chat_history[1:]) 

    messages_for_current_turn.append(HumanMessage(content=query))


   
    chat_completion = llm.invoke(messages_for_current_turn)

    ai_response_content = chat_completion.content
    print(f"🤖: {ai_response_content}")

    # Update chat history with the current human query and AI response
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=ai_response_content))