from rank_bm25 import BM25Okapi

class SparseRetriever:
    def __init__(self, documents):
        self.documents = documents
        tokenized = [doc.content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
    def query(self, query_text, top_k=7):
        tokens = query_text.lower().split()
        query_vector = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(query_vector)), key=lambda i: query_vector[i], reverse=True)[:top_k]
        return [(self.documents[i], query_vector[i]) for i in top_indices]