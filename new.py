from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


#step 1: Load PDF files
def load_pdf_file(pdf_path):
    """Loads a single PDF file and extracts text."""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Define the file path
pdf_path = "D:/Pr/Healt_care/data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
documents = load_pdf_file(pdf_path)
#print("Length of PDF pages: ", len(documents))

#step 2 : Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
# print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings 
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

def embed_in_batches(text_chunks, embedding_model, batch_size=64):
    db_main = None

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        try:
            db_batch = FAISS.from_documents(batch, embedding_model)
            if db_main is None:
                db_main = db_batch
            else:
                db_main.merge_from(db_batch)
        except Exception as e:
            print(f"Batch {i} failed: {e}")

    return db_main


 # Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db = embed_in_batches(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print("Embedding complete. Vector store saved.")