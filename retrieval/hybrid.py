def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    score = {}
    for rank, result in enumerate(dense_results):
        doc_id = result.payload["chunk_id"]
        score[doc_id] = score.get(doc_id, 0) + 1/ (k + rank + 1)
    
    for rank, (doc, _) in enumerate(sparse_results):
        doc_id = doc.metadata["chunk_id"]
        score[doc_id] = score.get(doc_id, 0) + 1/ (k + rank + 1)
        
    sorted_ids = sorted(score, key=score.get, reverse=True)
    return sorted_ids