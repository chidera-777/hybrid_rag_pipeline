# eval/faithfulness.py
from typing import Dict, List
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import *


class FaithfulnessEvaluator:
    """
    Evaluates whether generated answers are faithful to source documents.
    Detects hallucinations, contradictions, and unsupported claims.
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.0,  # Deterministic for evaluation
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing whether an AI's answer is faithful to provided sources.

            Your task is to identify:
            1. **Unsupported claims**: Statements not found in the sources
            2. **Contradictions**: Statements that conflict with the sources
            3. **Hallucinations**: Fabricated information presented as fact
            4. **Misrepresentations**: Correct information but misattributed or taken out of context

            Respond ONLY with valid JSON in this exact format:
            {{
                "faithful": true/false,
                "score": 0.0-1.0,
                "reasoning": "brief explanation of your assessment",
                "violations": ["specific issue 1", "specific issue 2", ...]
            }}

            If the answer is completely faithful, violations should be an empty array []."""),
            ("human", """Question: {question}

            Sources:
            {sources}

            Answer to Evaluate:
            {answer}

            Evaluate the faithfulness of this answer.""")
        ])
    
    def evaluate(self, question: str, answer: str, source_chunks: List[str]):
        sources = "\n\n".join([
            f"[SOURCE {i+1}]\n{chunk}"
            for i, chunk in enumerate(source_chunks)
        ])
        
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "question": question,
                "sources": sources,
                "answer": answer
            })
        
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if len(lines) > 2 else lines)
            
            result = json.loads(response)
            
            if not all(k in result for k in ["faithful", "score", "reasoning", "violations"]):
                raise ValueError("Missing required fields in response")
            
            return {
                "faithful": bool(result["faithful"]),
                "score": float(result["score"]),
                "reasoning": str(result["reasoning"]),
                "violations": list(result["violations"])
            }
        
        except json.JSONDecodeError as e:
            print(f"⚠ Failed to parse faithfulness evaluation: {e}")
            print(f"Raw response: {response[:200]}...")
            return {
                "faithful": False,
                "score": 0.0,
                "reasoning": "Evaluation failed: Invalid JSON response",
                "violations": ["Evaluator returned invalid format"]
            }
        
        except Exception as e:
            print(f"⚠ Faithfulness evaluation error: {e}")
            return {
                "faithful": False,
                "score": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "violations": ["Evaluation error"]
            }
