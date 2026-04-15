# Railway / Docker deployment
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling the CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Pre-download the embedding model so startup doesn't need HuggingFace access
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# Application code
COPY RAG.py .
COPY api_server.py .
COPY build_index.py .

# Pre-built FAISS index (run build_index_single.py or build_index.py locally first)
COPY index_file.index .
COPY index_file.meta.json .
COPY index_file.meta.chunks.json .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "api_server.py"]
