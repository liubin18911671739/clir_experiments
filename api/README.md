# CLIR Retrieval API

FastAPI-based REST API for cross-lingual information retrieval experiments.

## Features

- **BM25 Retrieval**: Traditional sparse retrieval with configurable parameters
- **Dense Retrieval**: mDPR-based neural retrieval
- **Hybrid Retrieval**: Multiple fusion strategies (RRF, weighted, CombSUM, CombMNZ)
- **Neural Reranking**: monoT5/mT5 reranking
- **Auto Documentation**: OpenAPI/Swagger UI at `/docs`
- **Docker Support**: Containerized deployment

## Quick Start

### Local Development

```bash
# Install API dependencies
pip install fastapi uvicorn[standard] pydantic

# Start development server
uvicorn api.main:app --reload --port 8000

# Access API documentation
open http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and start API
docker-compose up -d clir-api

# View logs
docker-compose logs -f clir-api

# Stop service
docker-compose down
```

### GPU Support

```bash
# Start GPU-enabled version
docker-compose --profile gpu up -d clir-api-gpu
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.5.0",
  "available_languages": ["fas", "rus", "zho"],
  "available_systems": ["bm25", "dense", "hybrid", "rerank"]
}
```

### BM25 Search

```bash
curl -X POST http://localhost:8000/search/bm25 \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning applications",
    "lang": "fas",
    "top_k": 10
  }'
```

**Response:**
```json
{
  "query": "machine learning applications",
  "lang": "fas",
  "num_results": 10,
  "results": [
    {
      "doc_id": "doc001",
      "score": 15.234,
      "rank": 1
    },
    ...
  ]
}
```

### Dense Search

```bash
curl -X POST http://localhost:8000/search/dense \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "lang": "fas",
    "top_k": 10
  }'
```

### Hybrid Search

```bash
# RRF fusion
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "information retrieval",
    "lang": "fas",
    "method": "rrf",
    "top_k": 10
  }'

# Weighted fusion (70% BM25, 30% Dense)
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "information retrieval",
    "lang": "fas",
    "method": "weighted",
    "alpha": 0.7,
    "top_k": 10
  }'
```

**Fusion Methods:**
- `rrf`: Reciprocal Rank Fusion
- `linear`: Linear combination
- `weighted`: Weighted fusion (controlled by `alpha`)
- `combsum`: Sum of normalized scores
- `combmnz`: CombSUM × number of non-zero scores

### Reranking

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "lang": "fas",
    "documents": [
      {"id": "doc001", "text": "Machine learning is a branch of AI..."},
      {"id": "doc002", "text": "Deep learning uses neural networks..."}
    ],
    "top_k": 10,
    "model": "monot5"
  }'
```

**Reranking Models:**
- `monot5`: Standard monoT5 model
- `mt5_multilingual`: Multilingual mT5 model

## Python Client Example

```python
import requests

API_BASE = "http://localhost:8000"

# BM25 Search
response = requests.post(
    f"{API_BASE}/search/bm25",
    json={
        "query": "machine learning",
        "lang": "fas",
        "top_k": 10
    }
)
results = response.json()
print(f"Found {results['num_results']} results")
for result in results['results']:
    print(f"Rank {result['rank']}: {result['doc_id']} (score: {result['score']:.3f})")

# Hybrid Search with RRF
response = requests.post(
    f"{API_BASE}/search/hybrid",
    json={
        "query": "neural networks",
        "lang": "fas",
        "method": "rrf",
        "top_k": 10
    }
)
results = response.json()
```

## Configuration

The API uses `config/neuclir.yaml` for all settings. Ensure:

1. **Indexes are built** before starting the API:
   ```bash
   python scripts/build_index_bm25.py --config config/neuclir.yaml --lang fas
   python scripts/build_index_dense.py --config config/neuclir.yaml --model mdpr --lang fas
   ```

2. **Config paths are correct**:
   - BM25 indexes: `indexes/bm25/{lang}/`
   - Dense indexes: `indexes/dense/{index_name}_{lang}/`

## Performance Tuning

### Caching

Searchers and models are cached on first use:
- BM25 searchers: Cached per language
- Dense searchers: Cached per language
- Rerankers: Cached per model

### Production Settings

```bash
# Run with multiple workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### GPU Acceleration

Enable GPU in `config/neuclir.yaml`:
```yaml
system:
  use_gpu: true
  gpu_device: 0
```

## API Documentation

Interactive documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid input parameters
- `404 Not Found`: Index or resource not found
- `500 Internal Server Error`: Processing error

**Example Error Response:**
```json
{
  "detail": "BM25 index not found for language 'fas' at /app/indexes/bm25/fas"
}
```

## Development

### Adding New Endpoints

1. Define Pydantic models for request/response
2. Implement endpoint function with type hints
3. Add error handling
4. Update documentation

### Testing

```bash
# Install test dependencies
pip install pytest httpx

# Run API tests
pytest tests/test_api.py -v
```

## Security Considerations

**For production deployment:**

1. **Add authentication**: Implement API key or OAuth
2. **Rate limiting**: Use middleware or reverse proxy
3. **HTTPS**: Deploy behind nginx with SSL
4. **Input validation**: Already handled by Pydantic
5. **CORS**: Configure allowed origins in production

## Troubleshooting

### Index Not Found

```
HTTPException: BM25 index not found for language 'fas'
```

**Solution**: Build indexes first:
```bash
python scripts/build_index_bm25.py --config config/neuclir.yaml --lang fas
```

### CUDA Out of Memory

**Solution**: Set `use_gpu: false` in config or reduce batch sizes

### Port Already in Use

**Solution**: Change port:
```bash
uvicorn api.main:app --port 8001
```

## License

MIT License - see LICENSE file for details.
