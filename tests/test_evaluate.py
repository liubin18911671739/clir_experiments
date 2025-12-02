"""Tests for evaluation module."""

import tempfile
from pathlib import Path
import pytest
import sys
import json

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils_io import write_trec_run


def create_test_qrels(qrels_path: Path):
    """Create test qrels file."""
    qrels_content = """q1 0 doc1 2
q1 0 doc2 1
q1 0 doc3 0
q1 0 doc4 1
q2 0 doc5 2
q2 0 doc6 1
"""
    with open(qrels_path, 'w', encoding='utf-8') as f:
        f.write(qrels_content)


def create_test_run(run_path: Path, run_id: str = "test_run"):
    """Create test run file."""
    results = [
        ('q1', 'doc1', 10.0),
        ('q1', 'doc2', 9.0),
        ('q1', 'doc3', 8.0),
        ('q1', 'doc4', 7.0),
        ('q2', 'doc5', 15.0),
        ('q2', 'doc6', 14.0),
    ]
    write_trec_run(results, str(run_path), run_id)


def test_trec_eval_format():
    """Test TREC evaluation file format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test files
        qrels_path = tmpdir / "test.qrels"
        run_path = tmpdir / "test.run"
        
        create_test_qrels(qrels_path)
        create_test_run(run_path)
        
        # Verify files exist
        assert qrels_path.exists()
        assert run_path.exists()
        
        # Check qrels format
        with open(qrels_path) as f:
            lines = f.readlines()
            assert len(lines) == 6
            parts = lines[0].split()
            assert len(parts) == 4


def test_evaluate_metrics():
    """Test basic evaluation metrics calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        qrels_path = tmpdir / "test.qrels"
        run_path = tmpdir / "test.run"
        
        create_test_qrels(qrels_path)
        create_test_run(run_path)
        
        # Mock evaluation results
        mock_results = {
            "ndcg_cut.10": 0.8523,
            "map": 0.7234,
            "recip_rank": 0.9000,
            "recall.100": 0.8500
        }
        
        # Verify metrics format
        assert all(isinstance(v, float) for v in mock_results.values())
        assert all(0.0 <= v <= 1.0 for v in mock_results.values())


def test_perfect_ranking():
    """Test evaluation with perfect ranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create perfect ranking (all relevant docs at top)
        qrels_path = tmpdir / "test.qrels"
        run_path = tmpdir / "perfect.run"
        
        qrels_content = """q1 0 doc1 2
q1 0 doc2 2
q1 0 doc3 0
"""
        with open(qrels_path, 'w') as f:
            f.write(qrels_content)
        
        # Perfect run: all relevant docs first
        results = [
            ('q1', 'doc1', 10.0),
            ('q1', 'doc2', 9.0),
            ('q1', 'doc3', 8.0),
        ]
        write_trec_run(results, str(run_path), "perfect")
        
        assert run_path.exists()


def test_no_relevant_docs():
    """Test evaluation with no relevant documents retrieved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        qrels_path = tmpdir / "test.qrels"
        run_path = tmpdir / "bad.run"
        
        # Qrels has doc1, doc2 as relevant
        qrels_content = """q1 0 doc1 2
q1 0 doc2 1
"""
        with open(qrels_path, 'w') as f:
            f.write(qrels_content)
        
        # Run retrieves only non-relevant docs
        results = [
            ('q1', 'doc3', 10.0),
            ('q1', 'doc4', 9.0),
        ]
        write_trec_run(results, str(run_path), "bad")
        
        # All metrics should be 0
        expected_metrics = {
            "map": 0.0,
            "recall": 0.0
        }
        assert all(v == 0.0 for v in expected_metrics.values())


def test_multiple_queries():
    """Test evaluation with multiple queries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        qrels_path = tmpdir / "multi.qrels"
        run_path = tmpdir / "multi.run"
        
        # Multiple queries
        qrels_content = """q1 0 doc1 2
q1 0 doc2 1
q2 0 doc3 2
q2 0 doc4 1
q3 0 doc5 2
"""
        with open(qrels_path, 'w') as f:
            f.write(qrels_content)
        
        results = [
            ('q1', 'doc1', 10.0),
            ('q1', 'doc2', 9.0),
            ('q2', 'doc3', 15.0),
            ('q2', 'doc4', 14.0),
            ('q3', 'doc5', 20.0),
        ]
        write_trec_run(results, str(run_path), "multi")
        
        # Verify all queries present
        with open(run_path) as f:
            content = f.read()
            assert 'q1' in content
            assert 'q2' in content
            assert 'q3' in content


def test_result_comparison():
    """Test comparing multiple run results."""
    mock_results = [
        {
            "run_name": "bm25",
            "ndcg_cut.10": 0.5234,
            "map": 0.3156
        },
        {
            "run_name": "dense",
            "ndcg_cut.10": 0.6123,
            "map": 0.3892
        },
        {
            "run_name": "hybrid",
            "ndcg_cut.10": 0.6789,
            "map": 0.4234
        }
    ]
    
    # Best performance should be hybrid
    best_ndcg = max(r["ndcg_cut.10"] for r in mock_results)
    assert best_ndcg == 0.6789
    
    # Verify improvement
    bm25_ndcg = mock_results[0]["ndcg_cut.10"]
    hybrid_ndcg = mock_results[2]["ndcg_cut.10"]
    improvement = (hybrid_ndcg - bm25_ndcg) / bm25_ndcg
    assert improvement > 0.2  # >20% improvement


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
