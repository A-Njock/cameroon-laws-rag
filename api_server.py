"""
FastAPI server for Cameroon Laws RAG System
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from RAG import RobustRAGSystem

# Initialize FastAPI
app = FastAPI(
    title="Cameroon Laws RAG API",
    description="Legal document retrieval and Q&A for Cameroon laws",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[RobustRAGSystem] = None


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


@app.on_event("startup")
async def startup_event():
    """Load RAG system on startup"""
    global rag_system
    
    index_path = "index_file.index"
    metadata_path = "index_file.meta.json"
    chunks_path = "index_file.meta.chunks.json"
    
    if not all(os.path.exists(p) for p in [index_path, metadata_path, chunks_path]):
        raise RuntimeError(
            "Index files not found. Run build_index.py first or check Docker CMD."
        )
    
    print("Loading RAG system...")
    rag_system = RobustRAGSystem.from_index(
        index_path=index_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        embedding_model='all-MiniLM-L6-v2',
        top_k=5
    )
    print(f"✓ RAG system loaded with {len(rag_system.chunks)} chunks")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Cameroon Laws RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return HealthResponse(
        status="healthy",
        documents_indexed=len(rag_system.chunks)
    )


@app.post("/query", response_model=QueryResponse)
async def query_laws(request: QueryRequest):
    """
    Query the Cameroon laws database
    
    Examples:
    - "Que dit l'article 7 de la loi N°2016-007?"
    - "Quelles sont les peines pour fraude?"
    - "Code pénal article 15"
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Generate answer
        answer = rag_system.generate_response(request.question)
        
        # Extract sources from metadata
        _, metadatas = rag_system.retrieve_for_generation(request.question)
        
        sources = []
        seen = set()
        for meta in metadatas[:request.top_k]:
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
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/laws", response_model=List[str])
async def list_laws():
    """List all available laws in the database"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Extract unique laws from metadata
    laws = set()
    for meta in rag_system.metadata:
        law = meta.get('law', 'Unknown')
        if law != 'Unknown':
            laws.add(law)
    
    return sorted(list(laws))


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
