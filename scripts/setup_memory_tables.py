"""
Script to create DynamoDB tables for agent memory storage.

Run this once to set up the required tables:
    python scripts/setup_memory_tables.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.memory.dynamodb_store import DynamoDBMemoryStore


def main():
    print("Creating DynamoDB tables for agent memory...")
    print("-" * 50)
    
    try:
        DynamoDBMemoryStore.create_tables(region="eu-west-1")
        print("-" * 50)
        print("✓ Tables created successfully!")
        print("\nTables created:")
        print("  - RAG-ConversationMemory (with 24-hour TTL)")
        print("  - RAG-PatternMemory")
        print("\nMemory persistence is now enabled.")
        return 0
    except Exception as e:
        print(f"✗ Error creating tables: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
