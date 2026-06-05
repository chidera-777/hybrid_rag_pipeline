from typing import Dict, List, Any, Optional
from collections import defaultdict
from agent.memory.base_memory import BaseMemory, MemoryEntry
from agent.memory.dynamodb_store import DynamoDBMemoryStore
import json


class PatternMemory(BaseMemory):
    """
    Long-term pattern memory with DynamoDB persistence.
    
    Purpose: Learn tenant-specific query patterns over time
    Scope: Persistent across conversations and service restarts
    Usage: Help agent navigate to relevant documents more efficiently
    
    Examples of patterns:
    - "This tenant's pricing queries often need enterprise tier docs"
    - "Technical questions usually require API documentation"
    - "Support queries typically involve troubleshooting section"
    
    Patterns are for NAVIGATION hints, NOT for generating facts.
    The agent still retrieves documents - patterns just help it retrieve better.
    """
    
    def __init__(self, tenant_id: str, max_patterns: int = 50, persist: bool = True):
        super().__init__(tenant_id)
        self.max_patterns = max_patterns
        self.patterns: List[MemoryEntry] = []
        self.query_categories = defaultdict(int)
        self.successful_retrievals = defaultdict(list)
        self.persist = persist
        self.db_store = DynamoDBMemoryStore() if persist else None
        
        # Load from DynamoDB on init
        if self.persist:
            self._load_from_db()
    
    def _load_from_db(self):
        """Load patterns from DynamoDB."""
        if not self.db_store:
            return
        
        patterns = self.db_store.get_patterns(self.tenant_id)
        for pattern in patterns:
            category = pattern['category']
            self.query_categories[category] = pattern.get('query_count', 0)
            self.successful_retrievals[category] = pattern.get('successful_docs', [])
            
            # Reconstruct pattern entry
            entry = MemoryEntry(
                content=f"Category: {category} | Query count: {pattern.get('query_count', 0)} | Sources: {', '.join(pattern.get('successful_docs', [])[:3])}",
                timestamp=str(pattern.get('last_updated', 0)),
                metadata={
                    "category": category,
                    "question_snippet": pattern.get('last_question', ''),
                    "successful_docs": pattern.get('successful_docs', [])
                }
            )
            self.patterns.append(entry)
    
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a pattern observation.
        
        Args:
            content: Pattern description
            metadata: Pattern metadata (category, confidence, etc.)
        """
        entry = MemoryEntry.create(content, metadata)
        self.patterns.append(entry)
        
        if len(self.patterns) > self.max_patterns:
            self.patterns = self.patterns[-self.max_patterns:]
    
    def record_query_pattern(self, question: str, category: str, successful_docs: List[str], metadata: Optional[Dict[str, Any]] = None):
        """
        Record a successful query pattern and persist to DynamoDB.
        
        Args:
            question: The question asked
            category: Query category (pricing, technical, support, etc.)
            successful_docs: Document sources that provided good answers
            metadata: Additional metadata
        """
        self.query_categories[category] += 1
        self.successful_retrievals[category].extend(successful_docs)
        
        if len(self.successful_retrievals[category]) > 20:
            self.successful_retrievals[category] = self.successful_retrievals[category][-20:]
        
        pattern_content = f"Category: {category} | Question type: {question[:50]}... | Successful sources: {', '.join(set(successful_docs[:3]))}"
        
        self.add(
            content=pattern_content,
            metadata={
                "category": category,
                "question_snippet": question[:100],
                "successful_docs": successful_docs,
                **(metadata or {})
            }
        )
        
        # Persist to DynamoDB
        if self.persist and self.db_store:
            self.db_store.save_pattern(
                tenant_id=self.tenant_id,
                category=category,
                question=question,
                successful_docs=successful_docs,
                metadata=metadata
            )
    
    def get_context(self, max_entries: int = 5):
        """
        Get pattern context for agent reasoning.
        
        Returns navigation hints based on learned patterns.
        """
        if not self.patterns and not self.query_categories:
            return "No learned patterns yet."
        
        context = "=== LEARNED PATTERNS ===\n\n"
        
        if self.query_categories:
            context += "Common query categories:\n"
            sorted_categories = sorted(
                self.query_categories.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for category, count in sorted_categories:
                context += f"  - {category}: {count} queries\n"
            context += "\n"
        
        if self.successful_retrievals:
            context += "Successful document sources by category:\n"
            for category, docs in list(self.successful_retrievals.items())[:3]:
                unique_docs = list(set(docs))[:3]
                context += f"  - {category}: {', '.join(unique_docs)}\n"
            context += "\n"
        
        if self.patterns:
            context += "Recent patterns:\n"
            for entry in self.patterns[-max_entries:]:
                context += f"  - {entry.content}\n"
            context += "\n"
        
        context += "Note: These patterns are HINTS for navigation, not facts for answers.\n"
        context += "Always retrieve fresh documents to answer questions.\n"
        
        return context
    
    def get_category_hints(self, question: str):
        """
        Get navigation hints by matching question keywords to learned patterns.
        
        Args:
            question: Current question
        
        Returns:
            Navigation hints or None
        """
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        hints = []
        for category, docs in self.successful_retrievals.items():
            category_words = set(category.lower().split())
            if question_words and category_words:
                unique_docs = list(set(docs))[:3]
                hints.append(f"{category.capitalize()} queries often found in: {', '.join(unique_docs)}")
        
        return "\n".join(hints[:3]) if hints else None
    
    def clear(self):
        """Clear all patterns from memory and DynamoDB."""
        self.patterns.clear()
        self.query_categories.clear()
        self.successful_retrievals.clear()
        
        if self.persist and self.db_store:
            self.db_store.clear_patterns(self.tenant_id)
    
    def get_all(self):
        """Get all pattern entries."""
        return self.patterns
    
    def get_statistics(self):
        """Get memory statistics."""
        return {
            "total_patterns": len(self.patterns),
            "query_categories": dict(self.query_categories),
            "categories_tracked": len(self.successful_retrievals),
            "most_common_category": max(self.query_categories.items(), key=lambda x: x[1])[0] if self.query_categories else None
        }
    
    def export(self):
        """Export patterns as JSON."""
        return json.dumps({
            "tenant_id": self.tenant_id,
            "patterns": [
                {
                    "timestamp": p.timestamp,
                    "content": p.content,
                    "metadata": p.metadata
                }
                for p in self.patterns
            ],
            "query_categories": dict(self.query_categories),
            "statistics": self.get_statistics()
        }, indent=2)
