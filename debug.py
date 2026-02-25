# debug_qdrant.py
from vectorstore.qdrant_store import QdrantStore
from config import *

store = QdrantStore(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME
)

# Query
results = store.query("What is RAG?", top_k=3)

print(f"Results type: {type(results)}")
print()

for i, result in enumerate(results):
    print(f"Result {i}:")
    print(f"  Type: {type(result)}")
    print(f"  Attributes: {dir(result)}")
    print(f"  Result object: {result}")
    print()
    
    # Try different ways to access payload
    if hasattr(result, 'payload'):
        print(f"  Has .payload: {result.payload}")
        print(f"  Payload type: {type(result.payload)}")
        
        # Try accessing chunk_id
        try:
            print(f"  chunk_id via ['chunk_id']: {result.payload['chunk_id']}")
        except:
            print(f"  chunk_id via ['chunk_id']: FAILED")
        
        try:
            print(f"  chunk_id via .get(): {result.payload.get('chunk_id')}")
        except:
            print(f"  chunk_id via .get(): FAILED")
    
    if hasattr(result, 'id'):
        print(f"  ID: {result.id}")
    
    if hasattr(result, 'score'):
        print(f"  Score: {result.score}")
    
    print("-" * 50)