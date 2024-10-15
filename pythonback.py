from flask import Flask, request, jsonify
import os
import openai
from pinecone import Pinecone, ServerlessSpec

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone using the new Pinecone class
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")  # Use your Pinecone API key from environment variables
)

# Create or use an existing Pinecone index
index_name = "powerapps-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # This dimension is for the text-embedding-ada-002 model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east1"  # Use the correct region for your deployment
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize Flask app
app = Flask(__name__)

# Endpoint to process a query from Power Automate
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Embed the query using OpenAI's embedding model
        query_embedding = openai.Embedding.create(input=user_query, engine="text-embedding-ada-002")['data'][0]['embedding']

        # Query Pinecone to retrieve relevant vectors
        pinecone_response = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        # Extract relevant text from the results
        relevant_text = " ".join([match['metadata']['text'] for match in pinecone_response['matches']])

        return jsonify({"relevant_text": relevant_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
