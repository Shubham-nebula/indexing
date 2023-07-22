from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone

app = Flask(__name__)

# Replace with your Pinecone API key
pinecone_api_key = "780e8927-3dd7-4d77-a97c-3d58aa7ae490"
pinecone_environment = "asia-southeast1-gcp-free"
index_name = "chatbot"

# Split documents into chunks
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Initialize Sentence Transformer for embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Split and index documents
def process_documents(pdf_directory):
    pdf_loader = PyPDFDirectoryLoader(pdf_directory)
    documents = pdf_loader.load()

    # Split and index the documents
    docs = split_docs(documents)
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    pinecone_index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    return "Documents processed and indexed successfully."

# Endpoint to handle incoming requests
@app.route('/process_documents', methods=['POST'])
def process_documents_route():
    try:
        data = request.get_json()
        pdf_directory = data.get('pdf_directory')

        if not pdf_directory:
            return jsonify({"error": "Invalid request. 'pdf_directory' key is missing."}), 400

        result = process_documents(pdf_directory)

        return jsonify({"message": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
