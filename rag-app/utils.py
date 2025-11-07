import os
import tempfile
from werkzeug.datastructures import FileStorage
from pdfplumber import PDF
from docx import Document as DocxDocument

def save_upload(uploaded_file: FileStorage, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    filename = uploaded_file.filename
    dest_path = os.path.join(dest_dir, filename)
    uploaded_file.save(dest_path)
    return dest_path

def extract_text_from_pdf(path):
    texts = []
    with PDF(open(path, "rb")) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)

def extract_text_from_docx(path):
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def extract_text(path):
    lower = path.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(path)
    if lower.endswith(".docx"):
        return extract_text_from_docx(path)
    if lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError("Unsupported file type")
