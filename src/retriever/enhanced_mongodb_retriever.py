

from typing import List, Dict, Any, Optional
import logging
from src.ingestion.mongodb_indexer import MongoDBVectorIndexer

logger = logging.getLogger(__name__)


class MongoDBRetriever:

    def __init__(self, embedder, vector_store_path: str = "data/embeddings/medical_practice_vectors"):
        self.indexer = MongoDBVectorIndexer(embedder, vector_store_path)
        self.indexer.load_index()
        logger.info(f"MongoDB retriever initialized with {len(self.indexer.documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        collection_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        results = self.indexer.search(query, top_k=top_k * 3)
        if collection_filter:
            results = [
                r for r in results
                if r['metadata'].get('collection') == collection_filter
            ]
            logger.info(f"Filtered to {len(results)} results from '{collection_filter}' collection")

        if filter_metadata:
            results = [
                r for r in results
                if all(r['metadata'].get(k) == v for k, v in filter_metadata.items())
            ]
            logger.info(f"Applied metadata filters, {len(results)} results remaining")

        return results[:top_k]
    
    def retrieve_by_collection(self, query: str, collection: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k=top_k, collection_filter=collection)
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 3,
        include_sources: bool = True,
        collection_filter: Optional[str] = None ) -> str:

        results = self.retrieve(query, top_k=top_k, collection_filter=collection_filter)
        
        if not results:
            return "No relevant information found in the database."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            if include_sources:
                collection_name = result['metadata'].get('collection', 'unknown')
                context_parts.append(f"[Document {i} - Source: {collection_name}]")
            else:
                context_parts.append(f"[Document {i}]")
            
            # Document text
            context_parts.append(result['text'])
            
            # Relevance score
            context_parts.append(f"(Relevance: {result['similarity_score']:.3f})")
            context_parts.append("")  # Empty line for readability
        
        formatted_context = "\n".join(context_parts)
        logger.info(f"Retrieved context from {len(results)} documents")
        
        return formatted_context
    
    def retrieve_context_with_metadata(
        self, 
        query: str, 
        top_k: int = 3,
        collection_filter: Optional[str] = None
    ) -> Dict[str, Any]:

        results = self.retrieve(query, top_k=top_k, collection_filter=collection_filter)
        
        # Build context string
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            collection_name = result['metadata'].get('collection', 'unknown')
            
            # Add to context
            context_parts.append(f"[Document {i} from {collection_name}]")
            context_parts.append(result['text'])
            context_parts.append("")
            
            # Add to sources list
            sources.append({
                'collection': collection_name,
                'text_preview': result['text'][:200] + "...",
                'score': result['similarity_score'],
                'metadata': result['metadata']
            })
        
        return {
            'context': "\n".join(context_parts),
            'sources': sources,
            'total_results': len(results)
        }
    
    def get_index_stats(self) -> Dict[str, Any]:

        collection_counts = {}
        for doc in self.indexer.documents:
            collection = doc['metadata'].get('collection', 'unknown')
            collection_counts[collection] = collection_counts.get(collection, 0) + 1
        
        return {
            'total_documents': len(self.indexer.documents),
            'collections': collection_counts,
            'available_collections': list(collection_counts.keys())
        }


# Factory function for easy initialization
def get_retriever(embedder, vector_store_path: str = "data/embeddings/medical_practice_vectors") -> MongoDBRetriever:

    return MongoDBRetriever(embedder, vector_store_path)