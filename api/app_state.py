_app_state = None

def set_app_state(embedder, reranker, generator):
    """Set app state for all endpoint modules."""
    global _app_state
    _app_state = {
        'embedder': embedder,
        'reranker': reranker,
        'generator': generator
    }

def get_app_state():
    """Get app state."""
    return _app_state
