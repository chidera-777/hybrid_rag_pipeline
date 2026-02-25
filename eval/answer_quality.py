# eval/answer_quality.py
from typing import Dict, Optional
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import *


class AnswerQualityEvaluator:
    """
    Evaluates overall answer quality: completeness, clarity, correctness.
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        
        # with no ground truth
        self.prompt_no_gt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing the quality of AI-generated answers.

            Rate the answer on three dimensions (0.0-1.0 scale):

            1. **Completeness** (0.0-1.0):
            - 1.0: Fully addresses all aspects of the question
            - 0.7-0.9: Addresses most aspects, minor gaps
            - 0.4-0.6: Addresses some aspects, significant gaps
            - 0.0-0.3: Barely addresses the question

            2. **Clarity** (0.0-1.0):
            - 1.0: Exceptionally clear, well-structured, easy to understand
            - 0.7-0.9: Clear and organized
            - 0.4-0.6: Somewhat clear but could be better organized
            - 0.0-0.3: Confusing or poorly structured

            3. **Correctness** (0.0-1.0):
            - 1.0: Information appears accurate and consistent
            - 0.7-0.9: Mostly accurate with minor issues
            - 0.4-0.6: Mix of accurate and questionable information
            - 0.0-0.3: Contains obvious errors or inconsistencies

            Respond ONLY with valid JSON:
            {{
                "completeness": 0.85,
                "clarity": 0.90,
                "correctness": 0.80,
                "feedback": "brief explanation of ratings"
            }}"""),
            ("human", """Question: {question}

            Answer:
            {answer}

            Evaluate the quality of this answer.""")
        ])
        
        # Prompt with ground truth
        self.prompt_with_gt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing the quality of AI-generated answers against ground truth.

            Rate the answer on three dimensions (0.0-1.0 scale):

            1. **Completeness**: How fully does it address the question compared to ground truth?
            2. **Clarity**: How well-structured and understandable is it?
            3. **Correctness**: How accurate is it compared to ground truth?

            Respond ONLY with valid JSON:
            {{
                "completeness": 0.85,
                "clarity": 0.90,
                "correctness": 0.80,
                "feedback": "brief explanation comparing to ground truth"
            }}"""),
            ("human", """Question: {question}

            Ground Truth Answer:
            {ground_truth}

            Generated Answer:
            {answer}

            Evaluate the quality of the generated answer.""")
        ])
    
    def evaluate(self, question: str, answer: str, ground_truth: Optional[str] = None) -> Dict:
        """
        Evaluate answer quality.
        
        Args:
            question: The original question
            answer: The generated answer
            ground_truth: Optional reference answer
        
        Returns:
            {
                "score": float (0-1, average of all metrics),
                "completeness": float (0-1),
                "clarity": float (0-1),
                "correctness": float (0-1),
                "feedback": str
            }
        """
        try:
            if ground_truth:
                prompt = self.prompt_with_gt
                input_vars = {
                    "question": question,
                    "answer": answer,
                    "ground_truth": ground_truth
                }
            else:
                prompt = self.prompt_no_gt
                input_vars = {
                    "question": question,
                    "answer": answer
                }
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke(input_vars)
            
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if len(lines) > 2 else lines)
            
            result = json.loads(response)
            
            score = (
                result["completeness"] + 
                result["clarity"] + 
                result["correctness"]
            ) / 3.0
            
            return {
                "score": float(score),
                "completeness": float(result["completeness"]),
                "clarity": float(result["clarity"]),
                "correctness": float(result["correctness"]),
                "feedback": str(result.get("feedback", ""))
            }
        
        except Exception as e:
            print(f"⚠ Answer quality evaluation error: {e}")
            return {
                "score": 0.0,
                "completeness": 0.0,
                "clarity": 0.0,
                "correctness": 0.0,
                "feedback": f"Evaluation failed: {str(e)}"
            }
