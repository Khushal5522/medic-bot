from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import time
from langchain.chains import ConversationalRetrievalChain

# Load .env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "mistral-saba-24b"
DB_FAISS_PATH ="vectorstore\db_faiss"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3002"],# Add more origins as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  
)

class QueryRequest(BaseModel):
    question: str

# Simple greeting responses
greetings = {
    "hi": "Hi! What medical question do you have?",
    "hello": "Hi! What medical question do you have?",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Goodbye! Have a great day!",
    "thank you": "You're welcome! Let me know if you have any other questions.",
    "thanks": "You're welcome! Let me know if you have any other questions.",
    "how are you": "I'm a bot, so I don't have feelings. How can I help you today?",
    "ok": "Okay! Let me know if you have a medical question.",
    "okay": "Alright! Feel free to ask a health-related query.",
    "cool": "Got it! I’m here to help with medical questions.",
}

# Emergency and help detection
def preprocess_user_query(user_input: str):
    text = user_input.strip().lower()

    emergency_keywords = ["chest pain", "can't breathe", "unconscious", "not breathing", "bleeding heavily"]
    if any(keyword in text for keyword in emergency_keywords):
        return "This sounds serious. Please seek immediate medical attention."

    help_requests = ["help", "i need help", "need help", "can you help me"]
    if text in help_requests:
        return "Sure, I’m here to help with health-related questions. Please share your symptoms or ask your medical query."

    non_medical_keywords = ["weather", "news", "sports", "movie", "music", "joke"]
    if any(keyword in text for keyword in non_medical_keywords):
        return "I’m only trained to help with health-related topics. Please ask a medical question."

    return None

# Load FAISS vector store
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        logging.error(f"Failed to load vector store: {e}")
        return None

with open("systemprompt.txt", "r", encoding="utf-8") as file:
    prompt_template_text = file.read()

custom_prompt = PromptTemplate(
    template=prompt_template_text,
    input_variables=["context", "question"]
)

# API endpoint
@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_input = request.question.strip().lower()

    # Greeting or special-case response
    if user_input in greetings:
        return {"answer": greetings[user_input]}

    handled_response = preprocess_user_query(user_input)
    if handled_response:
        return {"answer": handled_response}

    # Load vectorstore
    vectorstore = get_vectorstore()
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector store failed to load")

    # QA chain with Groq
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGroq(model_name=GROQ_MODEL, temperature=0.3, api_key=GROQ_API_KEY),
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            output_key="answer"  
        )

        # Retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = qa_chain.invoke({"question": request.question})
                return {"answer": response["answer"]}
            except Exception as e:
                logging.error(f"Groq call failed on attempt {attempt + 1}: {e}")
                time.sleep(1)  # short wait before retry

        return {"answer": "Sorry, the medical assistant is temporarily unavailable. Please try again shortly."}
    except Exception as e:
        logging.error(f"Unexpected error during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Root route
@app.get("/")
def read_root():
    return {"message": "Medic Bot API is running!"}
