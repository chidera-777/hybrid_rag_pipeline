import re
import logging
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from ingestion.base_loader import Document
from agent.prompts import REASONING_PROMPT, ANSWER_GENERATION_PROMPT
from agent.tools import ToolRegistry, ToolMode
from agent.memory import MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct agent for multi-step reasoning in RAG.
    
    Key principles:
    1. Reasoning is for navigation only (what to retrieve next)
    2. Final answer is grounded exclusively in accumulated observations
    3. Reasoning trace is NOT passed to answer generation
    """
    
    def __init__(
        self,
        rag_pipeline,
        llm: Optional[ChatGroq] = None,
        max_iterations: int = 3,
        api_key: Optional[str] = None,
        tool_mode: ToolMode = ToolMode.STRICT,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_memory: bool = False
    ):
        self.rag_pipeline = rag_pipeline
        self.max_iterations = max_iterations
        self.llm = llm or ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            streaming=True
        )
        self.tool_registry = tool_registry or ToolRegistry(mode=tool_mode)
        self.tool_mode = tool_mode
        self.memory_manager = memory_manager
        self.enable_memory = enable_memory
        
    def run(self, question: str, return_metadata: bool = False, stream: bool = False):
        """
        Execute ReAct loop: Thought → Action → Observation → repeat
        Then generate final answer from accumulated observations only.
        
        Args:
            question: The question to answer
            return_metadata: Whether to return reasoning trace
            stream: If True, yields events in real-time. If False, returns complete result.
        
        Returns:
            If stream=False: Dict with answer, sources, iterations, etc.
            If stream=True: Generator yielding event dicts
        """
        if stream:
            return self._run_streaming(question)
        
        return self._run_non_streaming(question, return_metadata)
    
    def _run_non_streaming(self, question: str, return_metadata: bool = False):
        """
        Execute ReAct loop: Thought → Action → Observation → repeat
        Then generate final answer from accumulated observations only.
        """
        observations = []
        reasoning_trace = []
        all_chunks = []
        chunk_ids_seen = set()
        source_attribution = {"faithful": [], "unfaithful": []}
        
        history = ""
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"ReAct Iteration {iteration}/{self.max_iterations}")
            
            thought, action, action_input = self._reason(question, history)
            reasoning_trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "action_input": action_input
            })
            
            logger.info(f"Thought: {thought}")
            logger.info(f"Action: {action}({action_input})")
            
            if action == "finish":
                break
        
            tool_result = self.tool_registry.execute_tool(
                tool_name=action,
                input_data=action_input,
                context={"rag_pipeline": self.rag_pipeline}
            )
            
            if not tool_result.success:
                logger.warning(f"Tool execution failed: {tool_result.error}")
                history += f"\nIteration {iteration}:\n"
                history += f"Thought: {thought}\n"
                history += f"Action: {action}\n"
                history += f"Action Input: {action_input}\n"
                history += f"Observation: Error - {tool_result.error}\n"
                continue
        
            is_faithful = tool_result.metadata.get("faithful", True)
            if is_faithful:
                source_attribution["faithful"].append(action)
            else:
                source_attribution["unfaithful"].append(action)
            
            if action == "retrieve" and tool_result.metadata and "chunks" in tool_result.metadata:
                chunks_data = tool_result.metadata["chunks"]
                new_chunks = []
                for chunk_data in chunks_data:
                    chunk = Document(
                        content=chunk_data["content"],
                        metadata=chunk_data["metadata"]
                    )
                    chunk_id = chunk.metadata.get("chunk_id", chunk.content_hash)
                    if chunk_id not in chunk_ids_seen:
                        chunk_ids_seen.add(chunk_id)
                        new_chunks.append(chunk)
                        all_chunks.append(chunk)
                
                observations.append({
                    "iteration": iteration,
                    "tool": action,
                    "query": action_input,
                    "chunks": new_chunks,
                    "text": tool_result.output,
                    "faithful": is_faithful
                })
                
                logger.info(f"Observation: Retrieved {len(new_chunks)} new chunks")
            else:
                observations.append({
                    "iteration": iteration,
                    "tool": action,
                    "query": action_input,
                    "text": tool_result.output,
                    "faithful": is_faithful
                })
                logger.info(f"Observation: {action} executed successfully")
            
            history += f"\nIteration {iteration}:\n"
            history += f"Thought: {thought}\n"
            history += f"Action: {action}\n"
            history += f"Action Input: {action_input}\n"
            history += f"Observation: {tool_result.output[:200]}...\n"
        
        if not observations:
            return {
                "answer": "I couldn't retrieve any relevant information to answer your question.",
                "sources": [],
                "model": "llama-3.3-70b-versatile",
                "reasoning_trace": reasoning_trace if return_metadata else None,
                "iterations": len(reasoning_trace),
                "source_attribution": source_attribution
            }
        
        final_answer = self._generate_answer(question, observations, all_chunks, source_attribution)
        
        if self.enable_memory and self.memory_manager:
            answer_summary = self._summarize_answer(final_answer["answer"], observations)
            successful_docs = [chunk.metadata.get("source", "unknown") for chunk in all_chunks]
            category = self._categorize_question(question)
            self.memory_manager.record_interaction(
                question=question,
                answer_summary=answer_summary,
                category=category,
                successful_docs=successful_docs
            )
        
        result = {
            "answer": final_answer["answer"],
            "sources": final_answer["sources"],
            "model": "llama-3.3-70b-versatile",
            "iterations": len(reasoning_trace),
            "source_attribution": source_attribution
        }
        
        if return_metadata:
            result["reasoning_trace"] = reasoning_trace
            result["total_chunks_retrieved"] = len(all_chunks)
            result["observations"] = [
                {
                    "iteration": obs["iteration"],
                    "tool": obs["tool"],
                    "query": obs["query"],
                    "faithful": obs["faithful"]
                }
                for obs in observations
            ]
            result["tool_mode"] = self.tool_mode.value
        
        return result
    
    def _run_streaming(self, question: str):
        """
        Execute ReAct loop with streaming output.
        Yields events for each reasoning step and answer generation.
        
        Yields:
            Dict events with types:
            - {"type": "reasoning", "iteration": 1, "thought": "...", "action": "...", "observation": "..."}
            - {"type": "answer_start", "model": "llama-3.3-70b"}
            - {"type": "answer_chunk", "content": "token"}
            - {"type": "answer_complete", "sources": [...], "iterations": 3}
        """
        observations = []
        all_chunks = []
        chunk_ids_seen = set()
        source_attribution = {"faithful": [], "unfaithful": []}
        history = ""
        iteration_count = 0
        
        for iteration in range(1, self.max_iterations + 1):
            iteration_count = iteration
            
            thought, action, action_input = self._reason(question, history)
            
            if action == "finish":
                break
            
            tool_result = self.tool_registry.execute_tool(
                tool_name=action,
                input_data=action_input,
                context={"rag_pipeline": self.rag_pipeline}
            )
            
            observation_text = tool_result.output if tool_result.success else f"Error: {tool_result.error}"
            
            # Stream reasoning step
            yield {
                "type": "reasoning",
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation_text[:200] + "..." if len(observation_text) > 200 else observation_text
            }
            
            if not tool_result.success:
                history += f"\nIteration {iteration}:\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: Error - {tool_result.error}\n"
                continue
            
            is_faithful = tool_result.metadata.get("faithful", True)
            if is_faithful:
                source_attribution["faithful"].append(action)
            else:
                source_attribution["unfaithful"].append(action)
            
            if action == "retrieve" and tool_result.metadata and "chunks" in tool_result.metadata:
                chunks_data = tool_result.metadata["chunks"]
                new_chunks = []
                for chunk_data in chunks_data:
                    chunk = Document(
                        content=chunk_data["content"],
                        metadata=chunk_data["metadata"]
                    )
                    chunk_id = chunk.metadata.get("chunk_id", chunk.content_hash)
                    if chunk_id not in chunk_ids_seen:
                        chunk_ids_seen.add(chunk_id)
                        new_chunks.append(chunk)
                        all_chunks.append(chunk)
                
                observations.append({
                    "iteration": iteration,
                    "tool": action,
                    "query": action_input,
                    "chunks": new_chunks,
                    "text": tool_result.output,
                    "faithful": is_faithful
                })
            else:
                observations.append({
                    "iteration": iteration,
                    "tool": action,
                    "query": action_input,
                    "text": tool_result.output,
                    "faithful": is_faithful
                })
            
            history += f"\nIteration {iteration}:\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {tool_result.output[:200]}...\n"
        
        if not observations:
            yield {
                "type": "answer_complete",
                "answer": "I couldn't retrieve any relevant information to answer your question.",
                "sources": [],
                "iterations": iteration_count
            }
            return
        
        # Stream answer generation
        yield {"type": "answer_start", "model": "llama-3.3-70b-versatile"}
        
        # Generate answer with streaming
        observations_text = self._build_observations_text(observations, all_chunks)
        
        chain = ANSWER_GENERATION_PROMPT | self.llm
        
        answer_chunks = []
        for chunk in chain.stream({
            "question": question,
            "observations": observations_text,
            "mode": self.tool_mode.value,
            "has_external": any(not obs.get("faithful", True) for obs in observations)
        }):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                answer_chunks.append(content)
                yield {"type": "answer_chunk", "content": content}
        
        full_answer = "".join(answer_chunks)
        
        if self.enable_memory and self.memory_manager:
            answer_summary = self._summarize_answer(full_answer, observations)
            successful_docs = [chunk.metadata.get("source", "unknown") for chunk in all_chunks]
            category = self._categorize_question(question)
            self.memory_manager.record_interaction(
                question=question,
                answer_summary=answer_summary,
                category=category,
                successful_docs=successful_docs
            )
        
        sources = [chunk.metadata for chunk in all_chunks]
        unfaithful_obs = [obs for obs in observations if not obs.get("faithful", True)]
        if unfaithful_obs:
            for obs in unfaithful_obs:
                sources.append({
                    "source": f"external_{obs['tool']}",
                    "tool": obs['tool'],
                    "faithful": False,
                    "query": obs['query']
                })
        
        # Final completion event
        yield {
            "type": "answer_complete",
            "sources": sources,
            "iterations": iteration_count,
            "source_attribution": source_attribution
        }
    
    def _build_observations_text(self, observations: List[Dict], all_chunks: List[Document]) -> str:
        """Build observations text for answer generation."""
        observations_text = ""
        source_counter = 1
        chunk_to_source = {}
        
        if all_chunks:
            observations_text += "=== KNOWLEDGE BASE (Faithful Sources) ===\n\n"
            for chunk in all_chunks:
                chunk_id = chunk.metadata.get("chunk_id", chunk.content_hash)
                if chunk_id not in chunk_to_source:
                    chunk_to_source[chunk_id] = source_counter
                    observations_text += f"[Source {source_counter}]\n{chunk.content}\n"
                    observations_text += f"[Metadata] {chunk.metadata}\n\n"
                    source_counter += 1
        
        unfaithful_obs = [obs for obs in observations if not obs.get("faithful", True)]
        if unfaithful_obs:
            observations_text += "\n=== EXTERNAL SOURCES (Unfaithful - Not from Knowledge Base) ===\n\n"
            for obs in unfaithful_obs:
                observations_text += f"[External Tool: {obs['tool']}]\n{obs['text']}\n\n"
        
        return observations_text
    
    def _reason(self, question: str, history: str):
        """
        Reasoning step: decide what tool to use next.
        Returns: (thought, action, action_input)
        """
        tools_description = self.tool_registry.get_tools_description()
        
        memory_context = ""
        if self.enable_memory and self.memory_manager:
            memory_context = self.memory_manager.get_navigation_context(
                current_question=question,
                include_conversation=True,
                include_patterns=True
            )
        
        chain = REASONING_PROMPT | self.llm | StrOutputParser()
        response = chain.invoke({
            "question": question,
            "history": history,
            "max_iterations": self.max_iterations,
            "tools": tools_description,
            "memory_context": memory_context
        })
        
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", response)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?=\n|$)", response, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        action = action_match.group(1).strip().lower() if action_match else "finish"
        action_input = action_input_match.group(1).strip() if action_input_match else ""
        
        return thought, action, action_input

    def _categorize_question(self, question: str) -> str:
        """
        Categorize the question into a general category.
        Simple keyword-based categorization.
        
        Args:
            question: The question text
        
        Returns:
            Category string
        """
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['price', 'pricing', 'cost', 'fee', 'payment', 'subscription']):
            return 'pricing'
        elif any(word in q_lower for word in ['api', 'code', 'implement', 'integrate', 'technical', 'sdk', 'endpoint']):
            return 'technical'
        elif any(word in q_lower for word in ['support', 'help', 'issue', 'problem', 'error', 'troubleshoot', 'bug']):
            return 'support'
        elif any(word in q_lower for word in ['feature', 'capability', 'can i', 'does it', 'functionality']):
            return 'features'
        elif any(word in q_lower for word in ['how to', 'tutorial', 'guide', 'setup', 'configure', 'install']):
            return 'howto'
        elif any(word in q_lower for word in ['policy', 'legal', 'terms', 'privacy', 'compliance', 'gdpr']):
            return 'policy'
        else:
            return 'general'
    
    def _summarize_answer(self, answer: str, observations: List[Dict]) -> str:
        """
        Generate a concise summary of the answer for memory storage.
        
        Args:
            answer: Full answer text
            observations: List of observations from ReAct loop
        
        Returns:
            Brief summary (max 150 chars)
        """
        first_sentence = answer.split('.')[0] if '.' in answer else answer
        
        if len(first_sentence) <= 150:
            summary = first_sentence.strip()
        else:
            summary = answer[:147].strip() + "..."
        
        tools_used = list(set([obs["tool"] for obs in observations]))
        if len(tools_used) > 1:
            summary += f" (via {', '.join(tools_used)})"
        
        return summary[:200]

    
    def _generate_answer(self, question: str, observations: List[Dict], all_chunks: List[Document], source_attribution: Dict[str, List[str]]):
        """
        Generate final answer from accumulated observations ONLY.
        The reasoning trace is NOT passed here - only the retrieved content.
        
        In RELAXED mode, clearly distinguish between faithful and unfaithful sources.
        """
        observations_text = ""
        source_counter = 1
        chunk_to_source = {}
        
        if all_chunks:
            observations_text += "=== KNOWLEDGE BASE (Faithful Sources) ===\n\n"
            for chunk in all_chunks:
                chunk_id = chunk.metadata.get("chunk_id", chunk.content_hash)
                if chunk_id not in chunk_to_source:
                    chunk_to_source[chunk_id] = source_counter
                    observations_text += f"[Source {source_counter}]\n{chunk.content}\n"
                    observations_text += f"[Metadata] {chunk.metadata}\n\n"
                    source_counter += 1
        
        unfaithful_obs = [obs for obs in observations if not obs.get("faithful", True)]
        if unfaithful_obs:
            observations_text += "\n=== EXTERNAL SOURCES (Unfaithful - Not from Knowledge Base) ===\n\n"
            for obs in unfaithful_obs:
                observations_text += f"[External Tool: {obs['tool']}]\n{obs['text']}\n\n"
        
        chain = ANSWER_GENERATION_PROMPT | self.llm | StrOutputParser()
        answer_text = chain.invoke({
            "question": question,
            "observations": observations_text,
            "mode": self.tool_mode.value,
            "has_external": len(unfaithful_obs) > 0
        })
        
        sources = [chunk.metadata for chunk in all_chunks]
        if unfaithful_obs:
            for obs in unfaithful_obs:
                sources.append({
                    "source": f"external_{obs['tool']}",
                    "tool": obs['tool'],
                    "faithful": False,
                    "query": obs['query']
                })
        
        return {
            "answer": answer_text,
            "sources": sources
        }
