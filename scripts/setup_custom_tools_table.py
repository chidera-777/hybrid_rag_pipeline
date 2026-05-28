import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.tools.tool_store import ToolStore

if __name__ == "__main__":
    print("Creating RAG-CustomTools table...")
    ToolStore.create_table()
    print("✅ Done!")
