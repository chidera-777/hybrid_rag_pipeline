from typing import Dict, List, Any, Optional
from collections import deque
from agent.memory.base_memory import BaseMemory, MemoryEntry
from agent.memory.dynamodb_store import DynamoDBMemoryStore


class ConversationMemory(BaseMemory):
    """
    Short-term conversation memory with DynamoDB persistence.
    
    Purpose: Maintain context across multi-turn conversations
    Scope: Current conversation only (24-hour TTL)
    Usage: Help agent understand follow-up questions
    
    Example:
    User: "What is the refund policy?"
    Agent: [retrieves and answers]
    User: "What about exchanges?" <- Memory helps understand "exchanges" relates to previous refund context
    """
    
    def __init__(self, tenant_id: str, conversation_id: str, max_turns: int = 10, persist: bool = True):
        super().__init__(tenant_id)
        self.conversation_id = conversation_id
        self.max_turns = max_turns
        self.conversation_history = deque(maxlen=max_turns)
        self.persist = persist
        self.db_store = DynamoDBMemoryStore() if persist else None
        if self.persist:
            self._load_from_db()
    
    def _load_from_db(self):
        """Load conversation history from DynamoDB."""
        if not self.db_store:
            return
        
        items = self.db_store.get_conversation_history(self.tenant_id, self.conversation_id, self.max_turns)
        for item in items:
            entry = MemoryEntry(
                content=f"Q: {item['question']}\nA: {item['answer_summary']}",
                timestamp=item['timestamp'],
                metadata={"type": "turn", "question": item['question']}
            )
            self.conversation_history.append(entry)
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a conversation turn.
        
        Args:
            content: The conversation turn (question or answer summary)
            metadata: Optional metadata (role, timestamp, etc.)
        """
        entry = MemoryEntry.create(content, metadata)
        self.conversation_history.append(entry)
    
    def add_turn(self, question: str, answer_summary: str):
        """
        Add a complete conversation turn and persist to DynamoDB.
        
        Args:
            question: User's question
            answer_summary: Brief summary of answer (NOT full answer)
        """
        self.add(
            content=f"Q: {question}\nA: {answer_summary}",
            metadata={"type": "turn", "question": question}
        )
        
        # Persist to DynamoDB
        if self.persist and self.db_store:
            self.db_store.save_conversation_turn(
                tenant_id=self.tenant_id,
                conversation_id=self.conversation_id,
                question=question,
                answer_summary=answer_summary
            )
    
    def get_context(self, max_entries: int = 5):
        """
        Get conversation context for agent reasoning.
        
        Returns formatted conversation history for navigation context.
        """
        if not self.conversation_history:
            return "No previous conversation context."
        
        recent_turns = list(self.conversation_history)[-max_entries:]
        
        context = "=== CONVERSATION CONTEXT (for navigation only) ===\n\n"
        for i, entry in enumerate(recent_turns, 1):
            context += f"Turn {i}:\n{entry.content}\n\n"
        
        context += "Note: Use this context to understand follow-up questions and navigate to relevant documents.\n"
        context += "Do NOT use this context to generate answers - always retrieve fresh documents.\n"
        
        return context
    
    def get_last_question(self):
        """Get the last question from conversation history."""
        if not self.conversation_history:
            return None
        
        last_entry = self.conversation_history[-1]
        return last_entry.metadata.get("question")
    
    def clear(self):
        """Clear conversation history from memory and DynamoDB."""
        self.conversation_history.clear()
        
        if self.persist and self.db_store:
            self.db_store.clear_conversation_history(self.tenant_id, self.conversation_id)
    
    def get_all(self):
        """Get all conversation entries."""
        return list(self.conversation_history)
    
    def get_summary(self):
        """Get a brief summary of the conversation."""
        if not self.conversation_history:
            return "No conversation history."
        
        return f"Conversation with {len(self.conversation_history)} turns. Last question: {self.get_last_question()}"
