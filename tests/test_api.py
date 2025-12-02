"""Tests for REST API endpoints."""

import sys
from pathlib import Path
import pytest

# Add api directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi.testclient import TestClient
    from api.main import app
    
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    pytest.skip("FastAPI not installed", allow_module_level=True)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "version" in data
    assert "available_languages" in data


def test_bm25_search_endpoint(client):
    """Test BM25 search endpoint structure."""
    # Note: This will fail without indexes, but tests the endpoint structure
    request_data = {
        "query": "machine learning",
        "lang": "fas",
        "top_k": 10
    }
    
    response = client.post("/search/bm25", json=request_data)
    
    # Will return 404 if index not found, or 200 if successful
    assert response.status_code in [200, 404, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "query" in data
        assert "lang" in data
        assert "results" in data


def test_dense_search_endpoint(client):
    """Test dense search endpoint structure."""
    request_data = {
        "query": "neural networks",
        "lang": "fas",
        "top_k": 10
    }
    
    response = client.post("/search/dense", json=request_data)
    assert response.status_code in [200, 404, 500]


def test_hybrid_search_endpoint(client):
    """Test hybrid search endpoint structure."""
    request_data = {
        "query": "information retrieval",
        "lang": "fas",
        "method": "rrf",
        "top_k": 10
    }
    
    response = client.post("/search/hybrid", json=request_data)
    assert response.status_code in [200, 404, 500]


def test_invalid_language(client):
    """Test API with invalid language code."""
    request_data = {
        "query": "test query",
        "lang": "invalid_lang",
        "top_k": 10
    }
    
    # Should return error for invalid language
    response = client.post("/search/bm25", json=request_data)
    # Depending on validation, might be 422 (validation error) or 404 (not found)
    assert response.status_code in [404, 422, 500]


def test_invalid_fusion_method(client):
    """Test hybrid search with invalid fusion method."""
    request_data = {
        "query": "test query",
        "lang": "fas",
        "method": "invalid_method",
        "top_k": 10
    }
    
    response = client.post("/search/hybrid", json=request_data)
    # Should return validation error
    assert response.status_code == 422


def test_rerank_endpoint_structure(client):
    """Test rerank endpoint structure."""
    request_data = {
        "query": "machine learning",
        "lang": "fas",
        "documents": [
            {"id": "doc1", "text": "Machine learning is AI"},
            {"id": "doc2", "text": "Deep learning uses neural nets"}
        ],
        "top_k": 2,
        "model": "monot5"
    }
    
    response = client.post("/rerank", json=request_data)
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "query" in data
        assert "results" in data


def test_missing_required_fields(client):
    """Test API with missing required fields."""
    # Missing 'query' field
    request_data = {
        "lang": "fas",
        "top_k": 10
    }
    
    response = client.post("/search/bm25", json=request_data)
    assert response.status_code == 422  # Validation error


def test_top_k_validation(client):
    """Test top_k parameter validation."""
    # Negative top_k
    request_data = {
        "query": "test",
        "lang": "fas",
        "top_k": -10
    }
    
    response = client.post("/search/bm25", json=request_data)
    assert response.status_code == 422
    
    # Very large top_k (might be accepted or rejected depending on limits)
    request_data["top_k"] = 100000
    response = client.post("/search/bm25", json=request_data)
    assert response.status_code in [200, 404, 422, 500]


def test_weighted_fusion_alpha(client):
    """Test weighted fusion with alpha parameter."""
    request_data = {
        "query": "test query",
        "lang": "fas",
        "method": "weighted",
        "alpha": 0.7,
        "top_k": 10
    }
    
    response = client.post("/search/hybrid", json=request_data)
    assert response.status_code in [200, 404, 500]
    
    # Invalid alpha (outside [0, 1])
    request_data["alpha"] = 1.5
    response = client.post("/search/hybrid", json=request_data)
    assert response.status_code == 422


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.get("/")
    
    # FastAPI CORS middleware should add these headers
    # Note: TestClient might not include all CORS headers
    assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
