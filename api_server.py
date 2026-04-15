"""
FastAPI server for Cameroon Laws RAG System
"""

# CRITICAL: Mock torchvision BEFORE any other imports to prevent transformers crash
import sys
import importlib.util
from types import ModuleType

# Create a proper mock module with __spec__ to pass importlib checks
class FakeTorchvision(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        self.__path__ = []
        self.__file__ = None
    def __getattr__(self, name):
        return FakeTorchvision(f"{self.__name__}.{name}")

# Install fake torchvision before anything else imports it
sys.modules["torchvision"] = FakeTorchvision("torchvision")
sys.modules["torchvision.ops"] = FakeTorchvision("torchvision.ops")
sys.modules["torchvision.transforms"] = FakeTorchvision("torchvision.transforms")
sys.modules["torchvision.transforms.v2"] = FakeTorchvision("torchvision.transforms.v2")
sys.modules["torchvision.transforms.v2.functional"] = FakeTorchvision("torchvision.transforms.v2.functional")
sys.modules["torchvision.transforms.functional"] = FakeTorchvision("torchvision.transforms.functional")
sys.modules["torchvision._meta_registrations"] = FakeTorchvision("torchvision._meta_registrations")
sys.modules["torchvision.io"] = FakeTorchvision("torchvision.io")
sys.modules["torchvision.models"] = FakeTorchvision("torchvision.models")
sys.modules["torchvision.datasets"] = FakeTorchvision("torchvision.datasets")
sys.modules["torchvision.utils"] = FakeTorchvision("torchvision.utils")

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
    from RAG import EMBEDDING_MODEL
    rag_system = RobustRAGSystem.from_index(
        index_path=index_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        embedding_model=EMBEDDING_MODEL,
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
@app.post("/ask", response_model=QueryResponse)
async def query_laws_api(request: QueryRequest):
    """
    Query the Cameroon laws database (Synchronous)
    Supports both /query and /ask endpoints.
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Generate answer and get metadatas in ONE call
        # Note: we use request.question for /query and request.query for legacy /ask if needed
        # But we'll use a unified QueryRequest model for now.
        answer, metadatas = rag_system.generate_response(request.question)
        
        sources = []
        seen = set()
        for meta in metadatas:
            law = meta.get('law', 'Unknown')
            article = meta.get('article', 'Unknown')
            key = f"{law}_{article}"
            
            if key not in seen:
                sources.append({
                    "law": law,
                    "article": article,
                    "citation": f"{article} - {law}"
                })
                seen.add(key)
        
        return QueryResponse(
            answer=answer,
            sources=sources[:request.top_k]
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")



from fastapi.responses import StreamingResponse

@app.post("/stream")
async def stream_laws(request: QueryRequest):
    """
    Stream the legal analysis in real-time
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    def event_generator():
        try:
            for part in rag_system.generate_response_stream(request.question):
                yield part
        except Exception as e:
            yield f"\n\n[ERREUR]: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain")



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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
