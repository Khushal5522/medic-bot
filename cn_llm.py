import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



# Step 1: Setup LLM (Mistral with HuggingFace)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "mistral-saba-24b"


def load_llm(groq_model):
    llm = ChatGroq(
        model_name=groq_model, 
        temperature=0.3,
        api_key=GROQ_API_KEY  
    )
    return llm

llm = load_llm(GROQ_MODEL)


# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful, polite, and brief medical assistant.

Follow these rules:
- Only answer health-related questions.
- Always use the provided context.
- If no relevant context is found, respond: “Please consult a doctor for more accurate help.”
- If the user asks for help (e.g., "help", "i need help"), respond: “Sure, I’m here to help with health-related questions. Please share your symptoms or ask your medical query.”
- If the user describes an emergency (e.g., “chest pain”, “unconscious”), respond: “This sounds serious. Please seek immediate medical attention.”
- If the question is not health-related, respond: “I’m only trained to help with health-related topics. Please ask a medical question.”
- Be polite. Keep answers short, clear, and helpful.
- Never give long answers.

Context: {context}
Question: {question}

Start your answer directly. Do not include greetings or small talk.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(GROQ_MODEL),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])
