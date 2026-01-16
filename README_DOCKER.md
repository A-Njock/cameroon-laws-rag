# Cameroon Laws RAG API - Docker Setup

## Quick Start

### 1. Build and Run
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### 2. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Query example
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Que dit l'\''article 7 de la loi N°2016-007?"}'

# List all laws
curl http://localhost:8000/laws
```

### 3. View API Documentation
Open in browser: `http://localhost:8000/docs`

## API Endpoints

### POST /query
Query the legal database
```json
{
  "question": "Quelles sont les peines pour fraude?",
  "top_k": 5
}
```

### GET /health
Check system status

### GET /laws
List all available laws

## Integration with Kemet AI Chatbot

Add this to your chatbot backend:

```python
import requests

def query_cameroon_laws(question: str) -> str:
    response = requests.post(
        "http://localhost:8000/query",
        json={"question": question}
    )
    result = response.json()
    return result["answer"]
```

## Stopping the Service
```bash
docker-compose down
```

## Rebuilding After Changes
```bash
docker-compose up --build --force-recreate
```
