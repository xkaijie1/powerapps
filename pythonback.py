from flask import Flask, request, jsonify
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec  # Import necessary Pinecone classes

app = Flask(__name__)

# Initialize OpenAI API

# Create a Pinecone client using the new syntax
pc = Pinecone(
    api_key="2a53c623-d3cd-47da-aac7-b535f80d60af"
)

# Check if the index exists, if not, create it
if "powerapps-index" not in pc.list_indexes().names():
    pc.create_index(
        name="powerapps-index",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east1"
        )
    )

# Get the index for querying
index = pc.Index("powerapps-index")

# Endpoint to process a query from Power Automate
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get('query', '')

    # Embed the query using OpenAI's embedding model
    query_embedding = client.embeddings.create(input=user_query,
    engine="text-embedding-ada-002")['data'][0]['embedding']

    # Query Pinecone to retrieve relevant vectors
    pinecone_response = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # Extract relevant text from the results
    relevant_text = " ".join([match['metadata']['text'] for match in pinecone_response['matches']])

    return jsonify({"relevant_text": relevant_text}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)



