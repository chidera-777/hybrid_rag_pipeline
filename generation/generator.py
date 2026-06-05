import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
import dotenv

dotenv.load_dotenv()

class Generator:
    def __init__(self):
        self.prompt = None
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            streaming=True
        )
        
    def generate(self, query:str, chunks:list):
        context = self.build_context(chunks)
        prompt = self.build_prompt(query, context)
        if self.llm is not None:
            chain = prompt | self.llm | StrOutputParser()
            answer_text = chain.invoke({"context": context, "query": query})
            return {
                "answer": answer_text,
                "sources": [chunk.metadata for chunk in chunks],
                "model": "llama-3.3-70b-versatile"
            }
        else:
            raise RuntimeError("No LLM found to generate answer")
        
    def build_context(self, chunks:list):
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"[Source {i+1}]\n{chunk.content}\n"
            context += f"[Metadata] {chunk.metadata}\n\n"
        return context
    
    def build_prompt(self, query:str, context:str):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions strictly based on the retrieved context provided to you.
                ## Core Behavior
                - Base **90% or more** of every response directly on the provided context. Use your own general knowledge only to improve clarity, grammar, or to briefly fill minor gaps — never to introduce new facts or claims not found in the context.
                - If the context does not contain enough information to answer the query, explicitly say: "The provided context does not contain sufficient information to answer this question fully." Do not fabricate or assume facts.

                ## Citations
                - After every factual claim or sentence drawn from the context, add an inline citation in this format: [Source N], where N corresponds to the source number in the context provided.
                - If a claim draws from multiple sources, cite all of them: [Source 1][Source 3].
                - Do not cite your own general knowledge — only cite claims traceable to the context.

                ## Response Format
                - Answer in clear, concise prose.
                - If no context is provided at all, respond with: "I don't have any context to draw from to answer your question. Please ensure relevant documents have been retrieved.
             """),
            ("human", "Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        ])
        return self.prompt
        
    
