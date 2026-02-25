def reciprocal_rank_fusion(dense_results, sparse_results, k=10):
    score = {}
    for rank, item in enumerate(dense_results or []):
        doc_id = item.payload["chunk_id"]
        score[doc_id] = score.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, pair in enumerate(sparse_results or []):
        try:
            doc, _ = pair
        except Exception:
            continue
        doc_id = doc.metadata["chunk_id"]
        score[doc_id] = score.get(doc_id, 0) + 1/ (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)
