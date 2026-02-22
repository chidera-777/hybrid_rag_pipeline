from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import uuid

class QdrantStore:
    def __init__(self, collection_name="rag_documents", host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = collection_name
        self._create_collection()
        
    def _create_collection(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        except Exception as e:
            print(f"Collection already exists or error creating it: {e}")
            
    
    def get_existing_chunk_ids(self):
        existing_ids = []
        offset = None
        limit = 100
        
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=limit,
                with_payload=True,
                with_vector=False
            )
            
            if not records:
                break
            
            for record in records:
                chunk_id = record.payload.get("chunk_id")
                if chunk_id:
                    existing_ids.append(chunk_id)
                    
            if offset is None:
                break
        return existing_ids
    
            
    def add_documents(self, documents:list, skip_existing:bool=True, batch_size:int=100):
        if not documents:
            return {"added": 0, "skipped": 0, "total": 0}
        stats = {"added": 0, "skipped": 0, "total": len(documents)}
        
        existing_ids = set()
        if skip_existing:
            existing_ids = self.get_existing_chunk_ids()
            
        docs_to_add = []
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", doc.content_hash)
            if skip_existing and chunk_id in existing_ids:
                stats["skipped"] += 1
            else:
                docs_to_add.append(doc)
                
        if not docs_to_add:
            return stats
        
        for i in range(0, len(docs_to_add), batch_size):
            batch = docs_to_add[i:i+batch_size]
            texts = [doc.content for doc in batch]
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
            
            points = []
            for doc, embedding in zip(batch, embeddings):
                chunk_id = doc.metadata.get("chunk_id", doc.content_hash)
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        **doc.metadata,
                        "content": doc.content,
                        "chunk_id": chunk_id,
                        "content_hash": doc.content_hash
                    }
                )
                points.append(point)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            stats["added"] += len(batch)
            print(f"Uploaded batch {1//batch_size + 1}: {len(batch)} chunks")
        return stats
    
    
    def delete_by_source(self, source:str):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source)
                        )
                    ]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting documents by source {source}: {e}")
            return False
        
        
    def get_collection_stats(self):
        """Get statistics about the collection"""
        info = self.client.get_collection(self.collection_name)
        return {
            "total_chunks": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance_metric": info.config.params.vectors.distance
        }
    
    
    def query(self, query_text, top_k=10):
        query_vector = self.embedder.encode([query_text]).tolist()[0]
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return results
    
    
    def clear_collection(self):
        self.client.delete_collection(self.collection_name)
        self._create_collection()
