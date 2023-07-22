from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from azure.storage.blob import BlobServiceClient, BlobClient
import os

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

# Endpoint to handle incoming requests for processing documents
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

# Function to download file from Azure Blob Storage
def download_file_from_blob(container_name, blob_name):
    try:
        connection_string = "DefaultEndpointsProtocol=https;AccountName=azuretestshubham832458;AccountKey=2yEaP59qlgKVv6kEUCA5ARB4wdV3ZRoL2X9zjYCcIxOSYAG1CSBbBlAMPx3uBIe7ilQtSh7purEK+AStvFn8GA==;EndpointSuffix=core.windows.net"  # Replace with your Azure Blob Storage connection string
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        destination_path = f"transcripts/{blob_name}"  # Replace with the desired destination path
        
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)  # Create the destination directory if it doesn't exist
        
        with open(destination_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        
        print(f"File downloaded successfully: {destination_path}")
        return True
    except Exception as e:
        print(f"Error downloading file '{blob_name}': {str(e)}")
        return False

# Endpoint to handle incoming requests for downloading files from Azure Blob Storage
@app.route("/download", methods=["POST"])
def download_file():
    try:
        payload = request.json
        blob_name = payload.get("blob_name")
        
        if not blob_name:
            return jsonify({"message": "Blob name not provided."}), 400
        
        success = download_file_from_blob("transcript", blob_name)
        
        if success:
            return jsonify({"message": "File download completed successfully."})
        else:
            return jsonify({"message": "File download failed."}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the app on 0.0.0.0:8000
    app.run(host="0.0.0.0", port=8000, debug=True)
