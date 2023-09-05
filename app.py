from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
import qdrant_client
import os
from flask import Flask, request, json, jsonify
from dotenv import load_dotenv

load_dotenv()

client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_config = qdrant_client.http.models.VectorParams(
    size=384,
    distance=qdrant_client.http.models.Distance.COSINE
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", ",", "\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


app = Flask(__name__)


@app.route('/', methods=['POST'])
def llm():
    try:
        if request.method == 'POST':
            data = json.loads(request.data)
            query = data['query']
            namespace = data['namespace']

            vectorstore = Qdrant(
                client=client,
                collection_name=namespace,
                embeddings=embeddings
            )

            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            response = qa.run(query)
            return jsonify({'message': 'Success', 'response': response}), 200

    except Exception as e:
        return jsonify({'message': 'Error', 'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if request.method == 'POST':
            data = json.loads(request.data)
            namespace = data['namespace']
            raw_text = data['raw_text']

            client.delete_collection(namespace)

            client.recreate_collection(
                collection_name=namespace,
                vectors_config=collection_config
            )

            vectorstore = Qdrant(
                client=client,
                collection_name=namespace,
                embeddings=embeddings
            )

            texts = get_chunks(raw_text)
            vectorstore.add_texts(texts)

            return jsonify({'message': 'Success'}), 200

    except Exception as e:
        return jsonify({'message': 'Error', 'error': str(e)}), 500


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
