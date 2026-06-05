from typing import Dict, Optional, Any
from agent.memory.conversation_memory import ConversationMemory
from agent.memory.pattern_memory import PatternMemory
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Orchestrates short-term and long-term memory for the agent.
    
    Components:
    1. Conversation Memory: Short-term context for current conversation
    2. Pattern Memory: Long-term patterns learned over time
    
    CRITICAL PRINCIPLE:
    All memory is for NAVIGATION ONLY, never for answer generation.
    
    Memory helps the agent:
    - Understand follow-up questions (conversation memory)
    - Navigate to relevant documents efficiently (pattern memory)
    
    Memory does NOT:
    - Provide facts for answers
    - Replace document retrieval
    - Generate claims
    """
    
    def __init__(self, tenant_id: str, conversation_id: str, enable_conversation_memory: bool = True, enable_pattern_memory: bool = True, max_conversation_turns: int = 10, max_patterns: int = 50, persist: bool = True):
        self.tenant_id = tenant_id
        self.conversation_id = conversation_id
        self.enable_conversation_memory = enable_conversation_memory
        self.enable_pattern_memory = enable_pattern_memory
        
        self.conversation_memory = ConversationMemory(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            max_turns=max_conversation_turns,
            persist=persist
        ) if enable_conversation_memory else None
        
        self.pattern_memory = PatternMemory(
            tenant_id=tenant_id,
            max_patterns=max_patterns,
            persist=persist
        ) if enable_pattern_memory else None
    
    def get_navigation_context(self, current_question: str, include_conversation: bool = True, include_patterns: bool = True):
        """
        Get memory context for agent navigation.
        
        Args:
            current_question: The current question being asked
            include_conversation: Include conversation context
            include_patterns: Include learned patterns
        
        Returns:
            Formatted memory context for navigation
        """
        context_parts = []
        
        context_parts.append("=== MEMORY CONTEXT ===\n")
        context_parts.append("This memory is to help you navigate and retrieve documents.")
        context_parts.append("Do NOT use memory to generate answers. Always retrieve fresh documents.\n")
        
        if include_conversation and self.conversation_memory:
            conv_context = self.conversation_memory.get_context(max_entries=3)
            context_parts.append(conv_context)
        
        if include_patterns and self.pattern_memory:
            category_hints = self.pattern_memory.get_category_hints(current_question)
            if category_hints:
                context_parts.append("\n=== NAVIGATION HINTS ===")
                context_parts.append(category_hints)
            
            pattern_context = self.pattern_memory.get_context(max_entries=3)
            context_parts.append(f"\n{pattern_context}")
        
        if len(context_parts) == 3: 
            return "No memory context available."
        
        return "\n".join(context_parts)
    
    def record_interaction(self, question: str, answer_summary: str, category: Optional[str] = None, successful_docs: Optional[list] = None):
        """
        Record a completed interaction.
        
        Args:
            question: The question asked
            answer_summary: Brief summary of the answer (NOT full answer)
            category: Query category (pricing, technical, support, etc.)
            successful_docs: Document sources that provided good answers
        """
        if self.conversation_memory:
            self.conversation_memory.add_turn(question, answer_summary)
        
        if self.pattern_memory and category and successful_docs:
            self.pattern_memory.record_query_pattern(
                question=question,
                category=category,
                successful_docs=successful_docs
            )
    
    def clear_conversation(self):
        """Clear conversation memory (start new conversation)."""
        if self.conversation_memory:
            self.conversation_memory.clear()
            logger.info(f"Cleared conversation memory for tenant {self.tenant_id}")
    
    def clear_patterns(self):
        """Clear pattern memory (reset learned patterns)."""
        if self.pattern_memory:
            self.pattern_memory.clear()
            logger.info(f"Cleared pattern memory for tenant {self.tenant_id}")
    
    def clear_all(self):
        """Clear all memory."""
        self.clear_conversation()
        self.clear_patterns()
        logger.info(f"Cleared all memory for tenant {self.tenant_id}")
    
    def get_statistics(self):
        """Get memory statistics."""
        stats = {
            "tenant_id": self.tenant_id,
            "conversation_memory_enabled": self.enable_conversation_memory,
            "pattern_memory_enabled": self.enable_pattern_memory
        }
        
        if self.conversation_memory:
            stats["conversation_turns"] = len(self.conversation_memory.get_all())
            stats["conversation_summary"] = self.conversation_memory.get_summary()
        
        if self.pattern_memory:
            stats["pattern_statistics"] = self.pattern_memory.get_statistics()
        
        return stats
    
    def export_patterns(self):
        """Export learned patterns as JSON."""
        if self.pattern_memory:
            return self.pattern_memory.export()
        return None
