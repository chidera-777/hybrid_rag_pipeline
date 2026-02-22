import re
import pymupdf
from pathlib import Path
from .base_loader import Document, BaseLoader
from .chunker import Chunker

class PDFLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        self.chunker = Chunker(source_type="pdf")
    
    def load(self, source: str):
        raw_docs =  self.extract(str(source))
        cleaned_docs = [self.clean(doc) for doc in raw_docs]
        chunks = self.chunker.chunk(cleaned_docs)
        return chunks
    
    def extract(self, source: str):
        documents = []
        pdf = pymupdf.open(source)
        file_name = Path(source).stem
        
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            
            if not text.strip():
                continue
            
            documents.append(Document(
                content=text,
                metadata={
                    "source": file_name,
                    "file_path": source,
                    "doc_type": "pdf",
                    "page": page_num + 1,
                    "total_pages": len(pdf)
                }
            ))
            
        pdf.close()
        return documents
    
    
    
    def clean(self, doc: Document):
        text = doc.content

        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\narXiv:\S+\n', '\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        
        return Document(content=text, metadata=doc.metadata)
    
    
    def load_directory(self, directory: str):
        all_chunks = []
        pdf_files = list(Path(directory).glob("*pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory}.")
            return []

        for pdf_path in pdf_files:
            print(f"Loading PDF-{pdf_path.name}")
            chunks = self.load(str(pdf_path))
            all_chunks.extend(chunks)

        return all_chunks

