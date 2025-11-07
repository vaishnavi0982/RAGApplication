# rag_pipeline.py
import os
import google.generativeai as genai
from vector_database import faiss_db

# --- Setup Gemini ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Retrieve top-k relevant documents from FAISS ---
def retrieve_docs(query, k=3):
    return faiss_db.similarity_search(query, k=k)

# --- Combine the retrieved chunks into one context string ---
def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# --- Generate an answer using Gemini ---
def generate_rag_answer(query):
    docs = retrieve_docs(query)
    if not docs:
        return "I couldn't find any relevant information in the database."

    context = get_context(docs)
    prompt = f"""
You are a knowledgeable and concise assistant.
Use only the information provided in the context to answer the user's question.
If the answer is not present in the context, say "I don't know".

Question:
{query}

Context:
{context}

Answer:
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini API Error:", e)
        return "An error occurred while generating the response."


