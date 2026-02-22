from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple
import time

class Reranker:
    def __init__(self, model_name:str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        
    def rerank(self, query:str, chunks:List, top_k:int=5, return_scores:bool=False, threshold:float=None):
        if not chunks:
            return []
        
        top_k = min(top_k, len(chunks))
        pairs = [(query, chunks.content) for chunks in chunks]
        start = time.time()
        scores = self.model.predict(pairs)
        end = time.time()
        print(f"Cross-encoder took {end - start:.2f} seconds")
        
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        if threshold is not None:
            scored_chunks = [scored_chunks[i] for i in range(top_k) if scored_chunks[i][1] > threshold]
        if return_scores:
            return scored_chunks
        else:
            return [chunk for chunk, _ in scored_chunks]
        
    def rerank_with_diversity(self, query:str, chunks:List, top_k:int=5, diversity_weight:float=0.3):
        if not chunks:
            return []
        
        top_k = min(top_k, len(chunks))
        pairs = [(query, chunks.content) for chunks in chunks]
        scores = self.model.predict(pairs)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-5)
        
        selected = []
        remaining_indices = list(range(len(chunks)))
        for _ in range(top_k):
            best_idx = None
            best_score = float("-inf")
            for idx in remaining_indices:
                relevance = scores[idx]
                if selected:
                    max_similiarity = max(self.text_similiarity(chunks[idx].content, selected_chunk.content) for selected_chunk in selected)
                    diversity = 1 - max_similiarity
                else:
                    diversity = 1
                    
                mmr_score = ((1 - diversity_weight) * relevance + diversity_weight * diversity)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(chunks[best_idx])
            remaining_indices.remove(best_idx)
            
        return selected
        
        
    def text_similiarity(self, text1:str, text2:str):
        word1 = set(text1.lower().split())
        word2 = set(text2.lower().split())
        intersection = word1.intersection(word2)
        return float(len(intersection)) / (len(word1) + len(word2) - len(intersection))
    
    
    def batch_rerank(self, queries:List[str], chunks_per_query:List[List], top_k:int=5):
        all_results = []
        for query, chunk_list in zip(queries, chunks_per_query):
            results = self.rerank(query, chunk_list, top_k=top_k)
            all_results.append(results)
        return all_results