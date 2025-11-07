# vector_database.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PDFS_DIR = os.getenv("PDFS_DIR", "pdfs")
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH", "vectorstore/db_faiss")


# --- Create directories if missing ---
os.makedirs(PDFS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_DB_PATH), exist_ok=True)

def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    print("documents:", documents)
    return documents

def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load or create FAISS DB ---
def load_or_create_faiss():
    if os.path.exists(FAISS_DB_PATH):
        print("✅ Loading existing FAISS DB...")
        return FAISS.load_local(FAISS_DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
    else:
        print("⚙️ Creating new FAISS DB (empty)...")
        return FAISS.from_texts([""], get_embedding_model())
    
def add_file_to_index(file_path):
    """
    Takes a PDF file path, extracts text, chunks it,
    embeds the chunks, and adds them to the FAISS DB.
    """
    print(f"📄 Adding {file_path} to FAISS index...")
    documents = load_pdf(file_path)
    text_chunks = create_chunks(documents)

    # Load existing FAISS DB (already done globally)
    global faiss_db

    # Add new chunks to the FAISS index
    faiss_db.add_documents(text_chunks)

    # Save updated index
    faiss_db.save_local(FAISS_DB_PATH)

    print(f"✅ Added {len(text_chunks)} chunks from {os.path.basename(file_path)} to FAISS DB.")
    return len(text_chunks)



# ✅ This will be imported by rag_pipeline
faiss_db = load_or_create_faiss()
