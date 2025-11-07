import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

from utils import save_upload, extract_text
import vector_database as vdb
import rag_pipeline as rp

UPLOAD_DIR = os.getenv("PDFS_DIR", "pdfs")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__, static_folder="../frontend/build", static_url_path="/")
CORS(app)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "no selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        saved_path = os.path.join(UPLOAD_DIR, filename)
        file.save(saved_path)
        # Use vector_database helper to add file into FAISS index
        try:
            # For PDFs we use the loader inside vector_database.add_file_to_index
            chunks_added = vdb.add_file_to_index(saved_path)
            return jsonify({"status": "ok", "chunks_added": chunks_added}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "file type not allowed"}), 400

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    try:
        res = rp.generate_rag_answer(message)
        return jsonify({"response": res}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optionally serve the React app (if you build it and want Flask to serve static files)
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    build_dir = app.static_folder
    if path != "" and os.path.exists(os.path.join(build_dir, path)):
        return send_from_directory(build_dir, path)
    else:
        return send_from_directory(build_dir, "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
