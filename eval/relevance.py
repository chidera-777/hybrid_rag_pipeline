# eval/relevance.py
from typing import Dict, List
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import *


class RelevanceEvaluator:
    """
    Evaluates whether retrieved chunks are relevant to the question.
    Measures retrieval quality before generation.
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing whether retrieved document chunks are relevant to a question.

            For each chunk, determine:
            - Does it contain information that helps answer the question?
            - Is it directly related to the question topic?
            - Would a human find this useful for answering?

            Rate each chunk's relevance on a 0-1 scale:
            - 1.0: Directly answers the question or provides key information
            - 0.7-0.9: Highly relevant, provides useful context
            - 0.4-0.6: Somewhat relevant, tangentially related
            - 0.1-0.3: Barely relevant, contains related keywords only
            - 0.0: Completely irrelevant

            Respond ONLY with valid JSON:
            {{
                "chunk_scores": [0.8, 0.9, 0.3, ...],
                "avg_relevance": 0.67,
                "relevant_count": 2,
                "reasoning": "brief explanation"
            }}"""),
            ("human", """Question: {question}

            Retrieved Chunks:
            {chunks}

            Evaluate the relevance of these chunks.""")
        ])
    
    def evaluate(self, question: str, retrieved_chunks: List) -> Dict:
        """
        Evaluate retrieval relevance.
        
        Args:
            question: The user's question
            retrieved_chunks: List of Document objects retrieved
        
        Returns:
            {
                "avg_relevance": float (0-1),
                "chunk_scores": List[float],
                "relevant_count": int,
                "precision_at_k": float,
                "reasoning": str
            }
        """
        chunks_text = "\n\n".join([
            f"[CHUNK {i+1}]\n{chunk.content[:500]}..."
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "question": question,
                "chunks": chunks_text
            })
            
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if len(lines) > 2 else lines)
            
            result = json.loads(response)
            threshold = 0.5
            relevant_count = sum(1 for score in result["chunk_scores"] if score >= threshold)
            precision_at_k = relevant_count / len(result["chunk_scores"]) if result["chunk_scores"] else 0.0
            
            return {
                "avg_relevance": float(result["avg_relevance"]),
                "chunk_scores": [float(s) for s in result["chunk_scores"]],
                "relevant_count": int(result["relevant_count"]),
                "precision_at_k": float(precision_at_k),
                "reasoning": str(result.get("reasoning", ""))
            }
        
        except Exception as e:
            print(f"⚠ Relevance evaluation error: {e}")
            # Fallback to simple keyword scoring
            return self._fallback_relevance(question, retrieved_chunks)
    
    def _fallback_relevance(self, question: str, chunks: List) -> Dict:
        """Simple keyword-based fallback if LLM eval fails"""
        import numpy as np
        
        question_words = set(question.lower().split())
        scores = []
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            intersection = question_words & chunk_words
            union = question_words | chunk_words
            jaccard = len(intersection) / len(union) if union else 0
            scores.append(jaccard)
        
        avg_relevance = np.mean(scores) if scores else 0.0
        relevant_count = sum(1 for s in scores if s > 0.3)
        
        return {
            "avg_relevance": float(avg_relevance),
            "chunk_scores": scores,
            "relevant_count": relevant_count,
            "precision_at_k": relevant_count / len(scores) if scores else 0.0,
            "reasoning": "Fallback keyword-based scoring"
        }
