"""Tests for fusion strategies."""

import sys
from pathlib import Path
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_hybrid import (
    reciprocal_rank_fusion, 
    linear_combination,
    combsum,
    combmnz
)


def test_rrf_basic():
    """Test basic RRF fusion."""
    run1 = {
        'q1': [('doc1', 1, 10.0), ('doc2', 2, 9.0), ('doc3', 3, 8.0)]
    }
    run2 = {
        'q1': [('doc3', 1, 15.0), ('doc1', 2, 14.0), ('doc4', 3, 13.0)]
    }
    
    fused = reciprocal_rank_fusion([run1, run2], k=60)
    
    assert 'q1' in fused
    q1_docs = [docid for docid, _ in fused['q1']]
    
    # doc1 and doc3 appear in both runs, should rank higher
    assert 'doc1' in q1_docs
    assert 'doc3' in q1_docs


def test_rrf_with_different_k():
    """Test RRF with different k values."""
    run1 = {'q1': [('doc1', 1, 100.0), ('doc2', 2, 50.0)]}
    run2 = {'q1': [('doc2', 1, 100.0), ('doc1', 2, 50.0)]}
    
    # Small k gives more weight to top ranks
    fused_k10 = reciprocal_rank_fusion([run1, run2], k=10)
    fused_k100 = reciprocal_rank_fusion([run1, run2], k=100)
    
    assert 'q1' in fused_k10
    assert 'q1' in fused_k100
    
    # Both should have same docs but potentially different scores
    docs_k10 = set(docid for docid, _ in fused_k10['q1'])
    docs_k100 = set(docid for docid, _ in fused_k100['q1'])
    assert docs_k10 == docs_k100


def test_linear_combination_equal_weights():
    """Test linear combination with equal weights."""
    run1 = {'q1': [('doc1', 1, 100.0), ('doc2', 2, 50.0)]}
    run2 = {'q1': [('doc2', 1, 100.0), ('doc3', 2, 50.0)]}
    
    fused = linear_combination([run1, run2], weights=[0.5, 0.5])
    
    q1_docs = [docid for docid, _ in fused['q1']]
    
    # doc2 appears in both with good scores
    assert 'doc2' in q1_docs
    assert len(q1_docs) >= 2


def test_combsum():
    """Test CombSUM fusion."""
    run1 = {'q1': [('doc1', 1, 100.0), ('doc2', 2, 50.0)]}
    run2 = {'q1': [('doc2', 1, 100.0), ('doc3', 2, 50.0)]}
    
    fused = combsum([run1, run2])
    
    assert 'q1' in fused
    q1_docs = [docid for docid, _ in fused['q1']]
    
    # doc2 appears in both, should have highest combined score
    assert 'doc2' in q1_docs


def test_combmnz():
    """Test CombMNZ fusion."""
    run1 = {'q1': [('doc1', 1, 100.0), ('doc2', 2, 50.0), ('doc3', 3, 25.0)]}
    run2 = {'q1': [('doc2', 1, 100.0), ('doc3', 2, 50.0), ('doc4', 3, 25.0)]}
    
    fused = combmnz([run1, run2])
    
    assert 'q1' in fused
    q1_docs = [docid for docid, _ in fused['q1']]
    
    # doc2 and doc3 appear in both, should rank higher
    # CombMNZ multiplies by count, so they get bonus
    assert 'doc2' in q1_docs
    assert 'doc3' in q1_docs


def test_empty_run():
    """Test fusion with empty run."""
    run1 = {'q1': [('doc1', 1, 10.0)]}
    run2 = {'q1': []}
    
    fused = reciprocal_rank_fusion([run1, run2])
    
    # Should still work with one empty run
    assert 'q1' in fused
    assert len(fused['q1']) > 0


def test_no_overlap():
    """Test fusion with no overlapping documents."""
    run1 = {'q1': [('doc1', 1, 10.0), ('doc2', 2, 9.0)]}
    run2 = {'q1': [('doc3', 1, 10.0), ('doc4', 2, 9.0)]}
    
    fused = linear_combination([run1, run2])
    
    q1_docs = set(docid for docid, _ in fused['q1'])
    
    # All docs should appear
    assert len(q1_docs) == 4
    assert {'doc1', 'doc2', 'doc3', 'doc4'} == q1_docs


def test_single_run_fusion():
    """Test fusion with single run (should return same results)."""
    run1 = {'q1': [('doc1', 1, 10.0), ('doc2', 2, 9.0)]}
    
    fused = reciprocal_rank_fusion([run1])
    
    q1_docs = [docid for docid, _ in fused['q1']]
    original_docs = [docid for docid, rank, score in run1['q1']]
    
    # Order should be preserved
    assert q1_docs == original_docs


def test_three_way_fusion():
    """Test fusion with three systems."""
    run1 = {'q1': [('doc1', 1, 10.0), ('doc2', 2, 9.0)]}
    run2 = {'q1': [('doc2', 1, 10.0), ('doc3', 2, 9.0)]}
    run3 = {'q1': [('doc3', 1, 10.0), ('doc1', 2, 9.0)]}
    
    fused = reciprocal_rank_fusion([run1, run2, run3])
    
    q1_docs = [docid for docid, _ in fused['q1']]
    
    # All three docs appear somewhere in all runs
    assert len(q1_docs) == 3


def test_score_consistency():
    """Test that fusion produces consistent scores."""
    run1 = {'q1': [('doc1', 1, 100.0), ('doc2', 2, 50.0), ('doc3', 3, 25.0)]}
    run2 = {'q1': [('doc1', 1, 100.0), ('doc2', 2, 50.0), ('doc3', 3, 25.0)]}
    
    fused = linear_combination([run1, run2])
    
    scores = [score for _, score in fused['q1']]
    
    # Scores should be in descending order
    assert scores == sorted(scores, reverse=True)
    
    # No negative scores
    assert all(s >= 0 for s in scores)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
