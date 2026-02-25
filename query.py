import argparse
from pipeline import RAGPipeline
from config import *

def main():
    parser = argparse.ArgumentParser(description="Query the RAG Pipeline")
    parser.add_argument(
        "--query",
        type=str,
        help="The question to ask the pipeline"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask the pipeline"
    )
    parser.add_argument(
        "--use-diversity",
        action="store_true",
        help="Use diversity in the reranker"
    )
    args = parser.parse_args()
    
    q = args.query if args.query is not None else args.question
    if not q:
        parser.error("Provide a query as a positional argument or with --query")
    
    pipeline = RAGPipeline(use_diversity=args.use_diversity)
    result = pipeline.query(q, return_metadata=True)
    print("ANSWER:")
    print("="*60)
    print(result["answer"])
    print("\n" + "="*60)
    print("SOURCES:")
    print("="*60)
    for i, source in enumerate(result["sources"], 1):
        print(f"\n[{i}] {source.get('source', 'Unknown')}")
        if 'page' in source:
            print(f"Page: {source['page']}")
        if 'url' in source:
            print(f"URL: {source['url']}")
        
if __name__ == "__main__":
    main()
