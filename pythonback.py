from flask import Flask, request, jsonify
import openai
import pinecone
import os

app = Flask(__name__)

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone with the new Pinecone client class
pinecone_client = pinecone.PineconeClient(
    api_key="2a53c623-d3cd-47da-aac7-b535f80d60af"
)

# Define the Pinecone index
if "powerapps-index" not in pinecone_client.list_indexes():
    pinecone_client.create_index(name="powerapps-index", dimension=1536)  # Example dimension

index = pinecone_client.index("powerapps-index")

# Endpoint to process a query from Power Automate
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get('query', '')

    # Embed the query using OpenAI's embedding model
    query_embedding = openai.Embedding.create(
        input=user_query,
        engine="text-embedding-ada-002"
    )['data'][0]['embedding']

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
    app.run(debug=True)