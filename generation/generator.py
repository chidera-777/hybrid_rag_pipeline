import requests
import json


class Generator:
    def __init__(self, use_local:bool=False):
        self.use_local = use_local
        if not self.use_local:
            self.llm = LLM
            
    def generate(self, query:str, chunks:list):
        context = self.build_context(chunks)
        prompt = self.build_prompt(query, context)
        
        if self.use_local:
            return self.generate_ollama(prompt, chunks)
        else:
            return self.generate_llm(prompt, chunks)
        
    def build_context(self, chunks:list):
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"[Source {i+1}]\n{chunk.content}\n"
            context += f"[Metadata] {chunk.metadata}\n\n"
        return context
    
    def build_prompt(self, query:str, context:str):
        return """
            You are a friendly and helpful AI Assistant.
            You are only allowed to answer queries with the provided context given below.
            After every claim, you are to cite your claims like this: [Source 1].
            You are not allowed to use your own knowledge to answer any user query. If no context was provided to aid you answer the user's query, tell the user that you cannot help with the query.
            {context}
            Question: {query}
            Answer:
        """
        
    def generate_llm(self, prompt, chunks):
        pass
    
    def generate_ollama(self, prompt, chunks):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1:8b",
                "prompt": prompt,
                "stream": False
            }
        )
        return {
            "answer": response.json()["response"],
            "sources": [chunk.metadata for chunk in chunks],
            "model": "llama3.1:8b-local"
        }