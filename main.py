from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse

app = FastAPI()



import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever.mongodb_retriever import MongoDBRetriever
from src.ingestion.multi_collection_embedder import get_embedder
from src.llm.enhanced_generator import generate_llm_response


print("Initializing RAG system...")
embedder = get_embedder()

mongodb_retriever = MongoDBRetriever(
    embedder=embedder,
    vector_store_path="data/embeddings/medical_practice_vectors"
)

print("✓ System ready!\n")


def query_mongodb_rag(query: dict, session_id: str = "default") -> str:

    user_query = query['query']

    retrieved_docs = mongodb_retriever.retrieve(user_query, top_k=3)

    print(f"\n[Retrieved {len(retrieved_docs)} documents]")
    for i, doc in enumerate(retrieved_docs, 1):
        collection = doc['metadata'].get('collection', 'unknown')
        score = doc['similarity_score']
        print(f"  {i}. {collection} (score: {score:.3f})")

    response = generate_llm_response(query, retrieved_docs, session_id=session_id)
    return response


@app.post('/api/query/')
async def get_response( history: str, current_query: str ):

        query = {
            'history': history,
            'query': current_query
        }


        print(f'\n\n************************************************************')
        print(f'Previous History was : {history}')
        print(f'************************************************************\n\n')
        message = query_mongodb_rag(query)

        print(f"\n\n\nQuery: {query}\n\n\n")
        print("AI:", message)

        response = {
            'status': True,
            'statuscode': 200,
            'text': message
        }

        return JSONResponse(content=response, status_code=200)


