# eval/eval_suite.py
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
from tabulate import tabulate

from pipeline import RAGPipeline
from eval.faithfulness import FaithfulnessEvaluator
from eval.relevance import RelevanceEvaluator
from eval.answer_quality import AnswerQualityEvaluator
from config import *


class EvalSuite:
    """
    Complete evaluation framework for RAG pipeline.
    """
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        print("Initializing evaluators...")
        self.faithfulness_eval = FaithfulnessEvaluator()
        self.relevance_eval = RelevanceEvaluator()
        self.quality_eval = AnswerQualityEvaluator()
        print("✓ Evaluators ready\n")
    
    def load_test_set(self, filepath: str) -> List[Dict]:
        with open(filepath) as f:
            return json.load(f)
    
    def run_evaluation(self, test_queries: List[Dict], verbose: bool = True):
        results = []
        
        print(f"Running evaluation on {len(test_queries)} queries...\n")
        
        for i, test_case in enumerate(test_queries):
            if verbose:
                print(f"[{i+1}/{len(test_queries)}] {test_case['question'][:70]}...")
            
            eval_start = time.time()
            
            # Step 1: Run query through pipeline
            try:
                response = self.pipeline.query(
                    test_case["question"],
                    return_metadata=True
                )
            except Exception as e:
                print(f"  ✗ Pipeline error: {e}")
                results.append(self._create_error_result(test_case, str(e)))
                continue
            
            # Step 2: Extract source content for faithfulness check
            source_texts = []
            chunk_map = {c.metadata["chunk_id"]: c for c in self.pipeline.chunks}
            
            for source_meta in response["sources"]:
                chunk_id = source_meta.get("chunk_id")
                if chunk_id and chunk_id in chunk_map:
                    source_texts.append(chunk_map[chunk_id].content)
            
            if not source_texts:
                print(f"  ⚠ Warning: No source texts found for faithfulness check")
            
            # Step 3: Evaluate faithfulness
            if verbose:
                print(f"  Evaluating faithfulness...")
            try:
                faith_result = self.faithfulness_eval.evaluate(
                    question=test_case["question"],
                    answer=response["answer"],
                    source_chunks=source_texts
                )
            except Exception as e:
                print(f"  ⚠ Faithfulness eval failed: {e}")
                faith_result = {
                    "score": 0.0,
                    "faithful": False,
                    "violations": [str(e)],
                    "reasoning": "Evaluation failed"
                }
            
            # Step 4: Evaluate retrieval relevance
            if verbose:
                print(f"  Evaluating relevance...")
            try:
                # Get the chunks used in generation
                retrieved_chunks = [
                    chunk_map[meta["chunk_id"]]
                    for meta in response["sources"]
                    if meta.get("chunk_id") in chunk_map
                ]
                
                rel_result = self.relevance_eval.evaluate(
                    question=test_case["question"],
                    retrieved_chunks=retrieved_chunks
                )
            except Exception as e:
                print(f"  ⚠ Relevance eval failed: {e}")
                rel_result = {
                    "avg_relevance": 0.0,
                    "precision_at_k": 0.0,
                    "reasoning": str(e)
                }
            
            # Step 5: Evaluate answer quality
            if verbose:
                print(f"  Evaluating quality...")
            try:
                qual_result = self.quality_eval.evaluate(
                    question=test_case["question"],
                    answer=response["answer"],
                    ground_truth=test_case.get("ground_truth")
                )
            except Exception as e:
                print(f"  ⚠ Quality eval failed: {e}")
                qual_result = {
                    "score": 0.0,
                    "completeness": 0.0,
                    "clarity": 0.0,
                    "correctness": 0.0,
                    "feedback": str(e)
                }
            
            eval_time = time.time() - eval_start
            
            # Compile result
            result = {
                "question": test_case["question"],
                "answer": response["answer"],
                "faithfulness_score": faith_result["score"],
                "faithfulness_violations": faith_result["violations"],
                "faithfulness_reasoning": faith_result.get("reasoning", ""),
                "relevance_score": rel_result["avg_relevance"],
                "relevance_precision": rel_result.get("precision_at_k", 0.0),
                "quality_score": qual_result["score"],
                "quality_breakdown": {
                    "completeness": qual_result["completeness"],
                    "clarity": qual_result["clarity"],
                    "correctness": qual_result["correctness"]
                },
                "quality_feedback": qual_result.get("feedback", ""),
                "pipeline_latency": response["metadata"]["total_time"],
                "eval_latency": eval_time,
                "sources_used": len(response["sources"])
            }
            
            results.append(result)
            
            if verbose:
                print(f"  Faith: {faith_result['score']:.2f} | "
                      f"Rel: {rel_result['avg_relevance']:.2f} | "
                      f"Qual: {qual_result['score']:.2f} | "
                      f"Time: {result['pipeline_latency']:.2f}s\n")
        
        # Compute aggregate metrics
        aggregate = self._compute_aggregates(results)
        
        return {
            "aggregate": aggregate,
            "per_query": results
        }
    
    def _create_error_result(self, test_case: Dict, error: str) -> Dict:
        """Create a result entry for a failed query"""
        return {
            "question": test_case["question"],
            "answer": f"ERROR: {error}",
            "faithfulness_score": 0.0,
            "faithfulness_violations": [error],
            "faithfulness_reasoning": "Pipeline failed",
            "relevance_score": 0.0,
            "relevance_precision": 0.0,
            "quality_score": 0.0,
            "quality_breakdown": {
                "completeness": 0.0,
                "clarity": 0.0,
                "correctness": 0.0
            },
            "quality_feedback": error,
            "pipeline_latency": 0.0,
            "eval_latency": 0.0,
            "sources_used": 0
        }
    
    def _compute_aggregates(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics"""
        if not results:
            return {}
        
        return {
            "total_queries": len(results),
            "avg_faithfulness": np.mean([r["faithfulness_score"] for r in results]),
            "avg_relevance": np.mean([r["relevance_score"] for r in results]),
            "avg_quality": np.mean([r["quality_score"] for r in results]),
            "avg_completeness": np.mean([r["quality_breakdown"]["completeness"] for r in results]),
            "avg_clarity": np.mean([r["quality_breakdown"]["clarity"] for r in results]),
            "avg_correctness": np.mean([r["quality_breakdown"]["correctness"] for r in results]),
            "avg_pipeline_latency": np.mean([r["pipeline_latency"] for r in results]),
            "avg_eval_latency": np.mean([r["eval_latency"] for r in results]),
            "total_violations": sum(len(r["faithfulness_violations"]) for r in results),
            "pass_rate_faithfulness": sum(1 for r in results if r["faithfulness_score"] >= FAITHFULNESS_THRESHOLD) / len(results),
            "pass_rate_relevance": sum(1 for r in results if r["relevance_score"] >= RELEVANCE_THRESHOLD) / len(results)
        }
    
    def print_report(self, eval_results: Dict):
        """Print formatted evaluation report"""
        agg = eval_results["aggregate"]
        
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        
        # Aggregate metrics table
        table_data = [
            ["Total Queries", agg['total_queries']],
            ["", ""],
            ["Avg Faithfulness", f"{agg['avg_faithfulness']:.3f}"],
            ["Avg Relevance", f"{agg['avg_relevance']:.3f}"],
            ["Avg Quality", f"{agg['avg_quality']:.3f}"],
            ["  - Completeness", f"{agg['avg_completeness']:.3f}"],
            ["  - Clarity", f"{agg['avg_clarity']:.3f}"],
            ["  - Correctness", f"{agg['avg_correctness']:.3f}"],
            ["", ""],
            ["Pass Rate (Faithfulness)", f"{agg['pass_rate_faithfulness']*100:.1f}%"],
            ["Pass Rate (Relevance)", f"{agg['pass_rate_relevance']*100:.1f}%"],
            ["", ""],
            ["Avg Pipeline Latency", f"{agg['avg_pipeline_latency']:.2f}s"],
            ["Avg Eval Latency", f"{agg['avg_eval_latency']:.2f}s"],
            ["Total Faithfulness Violations", agg['total_violations']]
        ]
        
        print("\n" + tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
        
        # Per-query breakdown
        print("\n" + "="*80)
        print("PER-QUERY RESULTS")
        print("="*80 + "\n")
        
        for i, result in enumerate(eval_results["per_query"], 1):
            print(f"[{i}] {result['question']}")
            print(f"    Faithfulness: {result['faithfulness_score']:.3f}")
            print(f"    Relevance: {result['relevance_score']:.3f}")
            print(f"    Quality: {result['quality_score']:.3f} "
                  f"(Completeness: {result['quality_breakdown']['completeness']:.2f} "
                  f"Clarity: {result['quality_breakdown']['clarity']:.2f} "
                  f"Correctness: {result['quality_breakdown']['correctness']:.2f})")
            print(f"    Latency: {result['pipeline_latency']:.2f}s")
            
            if result["faithfulness_violations"]:
                print(f"    ⚠ Violations:")
                for v in result["faithfulness_violations"]:
                    print(f"      - {v}")
            print()
    
    def save_results(self, eval_results: Dict, filepath: str):
        """Save evaluation results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n✓ Results saved to: {filepath}")