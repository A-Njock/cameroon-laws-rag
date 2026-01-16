# Cameroon Laws RAG API

Legal document retrieval and Q&A system for Cameroon laws using FAISS + Sentence Transformers.

## Features
- 📚 54 Cameroon law documents indexed
- 🔍 Article-level retrieval with metadata
- 🤖 AI-powered legal Q&A (DeepSeek)
- 🌐 REST API with FastAPI
- 🐳 Docker deployment ready

## Quick Start (Docker)

```bash
docker compose up --build
```

API available at: `http://localhost:8000`

## API Endpoints

### POST /query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Que dit l'\''article 7 de la loi N°2016-007?"}'
```

### GET /laws
List all available laws

### GET /health
Health check

## Deployment on Coolify

1. Push to GitHub
2. Create new app in Coolify
3. Point to this repository
4. Coolify will auto-build and deploy
5. Access via your Coolify domain

## Environment Variables

- `DEEPSEEK_API_KEY` - DeepSeek API key (optional, has default)

## Tech Stack

- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS
- **LLM**: DeepSeek
- **API**: FastAPI
- **Deployment**: Docker

## License

MIT
