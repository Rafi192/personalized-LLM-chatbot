from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever.mongodb_retriever import MongoDBRetriever
from src.ingestion.multi_collection_embedder import get_embedder
from src.llm.enhanced_generator import generate_llm_response
from src.ingestion.ingest_multi_collection_mongodb import main


print("Initializing RAG system...")
embedder = get_embedder()

if os.path.exists("data/embeddings/medical_practice_vectors"):
    mongodb_retriever = MongoDBRetriever(
        embedder=embedder,
        vector_store_path="data/embeddings/medical_practice_vectors"
    )
else:
    os.makedirs('data/embeddings', exist_ok=True)
    main()
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
async def get_response( history: str = Form(...), 
                       current_query: str = Form(...) ):
        
    try:
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
    
    except Exception as e:
        response = {
            'status': False,
            'statuscode': 500,
            'text': str(e)
        }
        return JSONResponse(content=response, status_code=500)

@app.put('/api/update-data/')
async def update_db(want_refresh: bool):
    try:
        if(want_refresh):
            message = main()
            response = {
                'status': True,
                'status code': 200,
                'text' : message
            }
        else:
            response = {
                'status': True,
                'status code': 200,
                'text' : "For refresh set want_refresh 'True'."
            }
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        response = {
            'status': False,
            'statuscode': 500,
            'text': str(e)
        }
        return JSONResponse(content=response, status_code=500)

