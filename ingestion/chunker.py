from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_loader import Document

class Chunker:
    def __init__(self, source_type:str):
        if source_type == "pdf":
            self.splitter = RecursiveCharacterTextSplitter(
                separators=["\n", "\n\n", " ", ""],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
        elif source_type == "web":
            self.splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
            
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
       
        
    def chunk(self, document):
        chunks = []
        
        for doc in document:
            if not doc.content or len(doc.content.strip()) < 50:
                continue
            
            split_texts = self.splitter.split_text(doc.content)
            for i, text in enumerate(split_texts):
                if len(text.strip()) < 50:
                    continue
                
                chunks.append(
                    Document(
                        content=text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata['source']}_chunk_{i}",
                            "chunk_index": i,
                            "chunk_length": len(text)
                        }
                    )
                )
        return chunks