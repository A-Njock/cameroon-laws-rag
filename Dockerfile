# Railway / Docker deployment
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install all Python dependencies (requirements.txt uses --extra-index-url for CPU torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so startup doesn't need HuggingFace access
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# Application code
COPY RAG.py .
COPY api_server.py .

# Pre-built FAISS index + metadata (7356 chunks, built from LOIS_237_CLEAN/)
# law_graph.json maps base laws -> amending laws for amendment expansion
COPY index_file.index .
COPY index_file.meta.json .
COPY index_file.meta.chunks.json .
COPY law_graph.json .

EXPOSE 8000

CMD ["python", "api_server.py"]
