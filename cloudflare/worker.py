"""
Cloudflare Worker for Cameroon Laws RAG API
Loads pre-built FAISS index from R2 and serves FastAPI endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# Initialize FastAPI
app = FastAPI(
    title="Cameroon Laws RAG API",
    description="Legal document retrieval for Cameroon laws",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_system = None
embedding_model = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


async def load_index_from_r2():
    """Load FAISS index and metadata from Cloudflare R2"""
    global rag_system, embedding_model
    
    if rag_system is not None:
        return  # Already loaded
    
    # R2 credentials from environment
    r2_account_id = os.getenv("R2_ACCOUNT_ID")
    r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
    r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.getenv("R2_BUCKET", "cameroon-laws-index")
    
    # Load files from R2 (implement R2 client here)
    # For now, assume files are in local storage
    index_path = "index_file.index"
    metadata_path = "index_file.meta.json"
    chunks_path = "index_file.chunks.json"
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create RAG system object
    class SimpleRAG:
        def __init__(self, index, chunks, metadata, model):
            self.index = index
            self.chunks = chunks
            self.metadata = metadata
            self.model = model
            self.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "sk-941d68a6421e4c3cb2bb17f4e53d258a")
        
        def query(self, question: str, top_k: int = 5):
            # Generate query embedding
            query_emb = self.model.encode([question])[0]
            
            # Search FAISS
            distances, indices = self.index.search(
                np.array([query_emb]).astype('float32'), 
                top_k
            )
            
            # Get chunks and metadata
            retrieved_chunks = [self.chunks[i] for i in indices[0]]
            retrieved_meta = [self.metadata[i] for i in indices[0]]
            
            # Generate answer with DeepSeek
            answer = self._generate_answer(question, retrieved_chunks, retrieved_meta)
            
            # Format sources
            sources = []
            seen = set()
            for meta in retrieved_meta:
                law = meta.get('law', 'Unknown')
                article = meta.get('article', 'Unknown')
                key = f"{law}_{article}"
                if key not in seen:
                    sources.append({
                        "law": law,
                        "article": article,
                        "citation": f"{article} de {law}"
                    })
                    seen.add(key)
            
            return answer, sources
        
        def _generate_answer(self, question, chunks, metas):
            client = OpenAI(
                api_key=self.deepseek_key,
                base_url="https://api.deepseek.com/v1"
            )
            
            context = "\n\n".join([
                f"[{i+1}] {meta.get('law', 'Unknown')} - {meta.get('article', 'Unknown')}:\n{chunk}"
                for i, (chunk, meta) in enumerate(zip(chunks, metas))
            ])
            
            system_prompt = """Tu es un expert juridique camerounais. 
Réponds UNIQUEMENT en utilisant les articles fournis.
Écris une réponse claire et concise en français.
Cite toujours les articles exacts."""
            
            user_prompt = f"Question: {question}\n\nArticles:\n{context}\n\nRéponds:"
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Selon {metas[0].get('article', 'Unknown')} de {metas[0].get('law', 'Unknown')}:\n\n{chunks[0][:500]}..."
    
    rag_system = SimpleRAG(index, chunks, metadata, embedding_model)
    print(f"✓ RAG system loaded with {len(chunks)} chunks")


@app.on_event("startup")
async def startup():
    """Load index on startup"""
    await load_index_from_r2()


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Cameroon Laws RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not loaded")
    
    return HealthResponse(
        status="healthy",
        documents_indexed=len(rag_system.chunks)
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not loaded")
    
    try:
        answer, sources = rag_system.query(request.question, request.top_k)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/laws", response_model=List[str])
async def list_laws():
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not loaded")
    
    laws = set()
    for meta in rag_system.metadata:
        law = meta.get('law', 'Unknown')
        if law != 'Unknown':
            laws.add(law)
    
    return sorted(list(laws))
