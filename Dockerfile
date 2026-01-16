# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy application code
COPY RAG.py .
COPY build_index.py .
COPY api_server.py .

# Copy PDF files
COPY *.pdf ./

# Expose port
EXPOSE 8000

# Health check - Allow 15 minutes for index building on first run
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Build index if not exists, then run API
CMD if [ ! -f "index_file.index" ]; then python build_index.py; fi && python api_server.py
