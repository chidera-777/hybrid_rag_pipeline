# run_eval.py
import argparse
from datetime import datetime
from pipeline import RAGPipeline
from eval.eval_suite import EvalSuite
from config import *


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument(
        "--test-file",
        type=str,
        default=str(EVAL_DIR / "test_queries.json"),
        help="Path to test queries JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (default: auto-generated)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--use-diversity",
        action="store_true",
        help="Use diversity in the reranker"
    )
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(use_diversity=args.use_diversity)
    
    # Initialize eval suite
    eval_suite = EvalSuite(pipeline)
    
    # Load test queries
    print(f"Loading test queries from: {args.test_file}")
    test_queries = eval_suite.load_test_set(args.test_file)
    print(f"Loaded {len(test_queries)} test queries\n")
    
    # Run evaluation
    print("Running evaluation...")
    results = eval_suite.run_evaluation(test_queries, verbose=args.verbose)
    
    # Print report
    eval_suite.print_report(results)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = EVAL_DIR / f"eval_results_{timestamp}.json"
    
    eval_suite.save_results(results, str(output_path))


if __name__ == "__main__":
    main()