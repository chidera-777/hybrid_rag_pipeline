from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib


@dataclass
class Document:
    content: str
    metadata: dict
    
    @property
    def content_hash(self):
        hash_input = f"{self.metadata.get('source', '')}{self.content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    @property
    def chunk_id(self):
        if "chunk_id" in self.metadata:
            return self.metadata["chunk_id"]
        return self.content_hash

class BaseLoader(ABC):
    @abstractmethod
    def load(self, source:str) -> list[Document]:
        pass