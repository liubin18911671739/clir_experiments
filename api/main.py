#!/usr/bin/env python3
"""
FastAPI-based online retrieval service for CLIR experiments.

Provides REST endpoints for:
- BM25 retrieval
- Dense retrieval (mDPR, ColBERT)
- Hybrid retrieval (RRF, weighted fusion, CombSUM, CombMNZ)
- Neural reranking (monoT5/mT5)

Usage:
    # Start server
    uvicorn api.main:app --reload --port 8000

    # With production settings
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Literal
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add scripts directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

from utils_io import load_yaml, get_repo_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# Pydantic Models
# ========================================

class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str = Field(..., description="Search query text")
    lang: str = Field(..., description="Language code (e.g., fas, rus, zho)")
    top_k: int = Field(default=1000, ge=1, le=10000, description="Number of results to return")

    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning applications",
                "lang": "fas",
                "top_k": 100
            }
        }


class BM25SearchRequest(SearchRequest):
    """Request model for BM25 search."""
    k1: Optional[float] = Field(default=None, description="BM25 k1 parameter (defaults to config)")
    b: Optional[float] = Field(default=None, description="BM25 b parameter (defaults to config)")


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    query: str = Field(..., description="Search query text")
    lang: str = Field(..., description="Language code")
    method: Literal["rrf", "linear", "weighted", "combsum", "combmnz"] = Field(
        default="rrf",
        description="Fusion method"
    )
    alpha: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for weighted fusion (BM25 weight)"
    )
    top_k: int = Field(default=1000, ge=1, le=10000)


class RerankRequest(BaseModel):
    """Request model for reranking."""
    query: str = Field(..., description="Search query text")
    lang: str = Field(..., description="Language code")
    documents: List[Dict[str, str]] = Field(
        ...,
        description="List of documents with 'id' and 'text' fields"
    )
    top_k: int = Field(default=100, ge=1, le=1000, description="Number of results to rerank")
    model: Literal["monot5", "mt5_multilingual"] = Field(
        default="monot5",
        description="Reranking model to use"
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning",
                "lang": "fas",
                "documents": [
                    {"id": "doc001", "text": "Document text here..."},
                    {"id": "doc002", "text": "Another document..."}
                ],
                "top_k": 10,
                "model": "monot5"
            }
        }


class SearchResult(BaseModel):
    """Single search result."""
    doc_id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., description="Result rank")
    text: Optional[str] = Field(None, description="Document text (if available)")


class SearchResponse(BaseModel):
    """Response model for search operations."""
    query: str
    lang: str
    num_results: int
    results: List[SearchResult]

    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning",
                "lang": "fas",
                "num_results": 3,
                "results": [
                    {"doc_id": "doc001", "score": 15.234, "rank": 1},
                    {"doc_id": "doc005", "score": 14.891, "rank": 2},
                    {"doc_id": "doc012", "score": 13.456, "rank": 3}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    available_languages: List[str]
    available_systems: List[str]


# ========================================
# FastAPI Application
# ========================================

app = FastAPI(
    title="CLIR Retrieval API",
    description="Cross-lingual Information Retrieval API with BM25, Dense, Hybrid, and Reranking capabilities",
    version="2.5.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Global State
# ========================================

class APIState:
    """Global state for caching searchers and models."""
    def __init__(self):
        self.config = None
        self.bm25_searchers = {}  # {lang: searcher}
        self.dense_searchers = {}  # {lang: searcher}
        self.rerankers = {}  # {model_name: reranker}

    def load_config(self, config_path: str = "config/neuclir.yaml"):
        """Load configuration."""
        if self.config is None:
            repo_root = get_repo_root()
            full_path = repo_root / config_path
            self.config = load_yaml(str(full_path))
            logger.info(f"Loaded configuration from {full_path}")
        return self.config


state = APIState()


# ========================================
# Startup/Shutdown Events
# ========================================

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    logger.info("Starting CLIR Retrieval API...")
    try:
        state.load_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down CLIR Retrieval API...")


# ========================================
# Helper Functions
# ========================================

def get_bm25_searcher(lang: str, k1: Optional[float] = None, b: Optional[float] = None):
    """Get or create BM25 searcher for language."""
    from pyserini.search.lucene import LuceneSearcher

    cache_key = f"{lang}_{k1}_{b}"

    if cache_key not in state.bm25_searchers:
        config = state.load_config()
        repo_root = get_repo_root()

        bm25_config = config['bm25']
        index_dir = resolve_path(config['indexes']['bm25_dir'], repo_root)
        index_path = index_dir / lang

        if not index_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"BM25 index not found for language '{lang}' at {index_path}"
            )

        logger.info(f"Loading BM25 searcher for {lang} from {index_path}")
        searcher = LuceneSearcher(str(index_path))

        # Set BM25 parameters
        k1_param = k1 if k1 is not None else bm25_config['k1']
        b_param = b if b is not None else bm25_config['b']
        searcher.set_bm25(k1_param, b_param)

        state.bm25_searchers[cache_key] = searcher
        logger.info(f"BM25 searcher loaded for {lang} (k1={k1_param}, b={b_param})")

    return state.bm25_searchers[cache_key]


def get_dense_searcher(lang: str):
    """Get or create dense searcher for language."""
    from pyserini.search.faiss import FaissSearcher
    from pyserini.encode import AutoQueryEncoder

    if lang not in state.dense_searchers:
        config = state.load_config()
        repo_root = get_repo_root()

        mdpr_config = config['dense']['mdpr']
        index_dir = resolve_path(config['indexes']['dense_dir'], repo_root)
        index_name = mdpr_config['index_name']
        index_path = index_dir / f"{index_name}_{lang}"

        if not index_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Dense index not found for language '{lang}' at {index_path}"
            )

        logger.info(f"Loading dense searcher for {lang} from {index_path}")

        encoder = AutoQueryEncoder(
            model_name=mdpr_config['query_encoder'],
            pooling='cls',
            l2_norm=True,
            device='cuda' if config['system']['use_gpu'] else 'cpu'
        )

        searcher = FaissSearcher(str(index_path), encoder)
        state.dense_searchers[lang] = searcher
        logger.info(f"Dense searcher loaded for {lang}")

    return state.dense_searchers[lang]


# ========================================
# API Endpoints
# ========================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    config = state.load_config()

    return HealthResponse(
        status="healthy",
        version="2.5.0",
        available_languages=config['languages'],
        available_systems=["bm25", "dense", "hybrid", "rerank"]
    )


@app.post("/search/bm25", response_model=SearchResponse)
async def search_bm25(request: BM25SearchRequest):
    """
    Perform BM25 retrieval.

    Returns ranked list of document IDs with scores.
    """
    try:
        searcher = get_bm25_searcher(request.lang, request.k1, request.b)

        logger.info(f"BM25 search: query='{request.query}', lang={request.lang}, top_k={request.top_k}")
        hits = searcher.search(request.query, k=request.top_k)

        results = [
            SearchResult(
                doc_id=hit.docid,
                score=hit.score,
                rank=idx + 1
            )
            for idx, hit in enumerate(hits)
        ]

        return SearchResponse(
            query=request.query,
            lang=request.lang,
            num_results=len(results),
            results=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/dense", response_model=SearchResponse)
async def search_dense(request: SearchRequest):
    """
    Perform dense retrieval using mDPR.

    Returns ranked list of document IDs with scores.
    """
    try:
        searcher = get_dense_searcher(request.lang)

        logger.info(f"Dense search: query='{request.query}', lang={request.lang}, top_k={request.top_k}")
        hits = searcher.search(request.query, k=request.top_k)

        results = [
            SearchResult(
                doc_id=hit.docid,
                score=hit.score,
                rank=idx + 1
            )
            for idx, hit in enumerate(hits)
        ]

        return SearchResponse(
            query=request.query,
            lang=request.lang,
            num_results=len(results),
            results=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dense search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(request: HybridSearchRequest):
    """
    Perform hybrid retrieval combining BM25 and Dense results.

    Supports multiple fusion strategies:
    - rrf: Reciprocal Rank Fusion
    - linear: Linear combination
    - weighted: Weighted fusion (controlled by alpha)
    - combsum: Sum of normalized scores
    - combmnz: CombSUM × number of non-zero scores
    """
    try:
        # Get both BM25 and Dense results
        bm25_searcher = get_bm25_searcher(request.lang)
        dense_searcher = get_dense_searcher(request.lang)

        logger.info(f"Hybrid search ({request.method}): query='{request.query}', lang={request.lang}")

        bm25_hits = bm25_searcher.search(request.query, k=request.top_k)
        dense_hits = dense_searcher.search(request.query, k=request.top_k)

        # Convert to run format
        bm25_run = {request.query: [(hit.docid, hit.score) for hit in bm25_hits]}
        dense_run = {request.query: [(hit.docid, hit.score) for hit in dense_hits]}

        # Import fusion functions
        sys.path.insert(0, str(repo_root / "scripts"))
        from run_hybrid import (
            reciprocal_rank_fusion,
            linear_combination,
            weighted_fusion,
            combsum,
            combmnz
        )

        # Apply fusion
        if request.method == "rrf":
            fused = reciprocal_rank_fusion([bm25_run, dense_run])
        elif request.method == "linear":
            fused = linear_combination([bm25_run, dense_run])
        elif request.method == "weighted":
            fused = weighted_fusion([bm25_run, dense_run], request.alpha)
        elif request.method == "combsum":
            fused = combsum([bm25_run, dense_run])
        elif request.method == "combmnz":
            fused = combmnz([bm25_run, dense_run])
        else:
            raise HTTPException(status_code=400, detail=f"Unknown fusion method: {request.method}")

        # Convert to response format
        fused_docs = fused[request.query][:request.top_k]
        results = [
            SearchResult(
                doc_id=doc_id,
                score=score,
                rank=idx + 1
            )
            for idx, (doc_id, score) in enumerate(fused_docs)
        ]

        return SearchResponse(
            query=request.query,
            lang=request.lang,
            num_results=len(results),
            results=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/rerank", response_model=SearchResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents using monoT5 or mT5 models.

    Accepts a list of documents with the query and returns reranked results.
    """
    try:
        from rerank_mt5 import MonoT5Reranker

        # Get or create reranker
        if request.model not in state.rerankers:
            config = state.load_config()
            mt5_config = config['reranking']['mt5']

            model_name = (
                mt5_config['multilingual_model_name']
                if request.model == "mt5_multilingual"
                else mt5_config['model_name']
            )

            logger.info(f"Loading reranker: {model_name}")
            reranker = MonoT5Reranker(
                model_name=model_name,
                device='cuda' if config['system']['use_gpu'] else 'cpu',
                batch_size=mt5_config['batch_size']
            )
            state.rerankers[request.model] = reranker

        reranker = state.rerankers[request.model]

        # Prepare documents
        doc_ids = [doc['id'] for doc in request.documents]
        doc_texts = [doc['text'] for doc in request.documents]

        logger.info(f"Reranking {len(doc_ids)} documents for query: '{request.query}'")

        # Rerank
        reranked_scores = reranker.rerank_batch(
            request.query,
            doc_ids,
            doc_texts
        )

        # Sort by score and take top_k
        sorted_results = sorted(
            reranked_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:request.top_k]

        results = [
            SearchResult(
                doc_id=doc_id,
                score=score,
                rank=idx + 1
            )
            for idx, (doc_id, score) in enumerate(sorted_results)
        ]

        return SearchResponse(
            query=request.query,
            lang=request.lang,
            num_results=len(results),
            results=results
        )

    except Exception as e:
        logger.error(f"Reranking error: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
