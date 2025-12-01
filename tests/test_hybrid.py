"""Tests for hybrid retrieval module."""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_hybrid import reciprocal_rank_fusion, linear_combination


def test_reciprocal_rank_fusion():
    """Test RRF fusion."""
    # Create mock runs
    run1 = {
        'q1': [('doc1', 1, 10.0), ('doc2', 2, 9.0), ('doc3', 3, 8.0)],
        'q2': [('doc5', 1, 15.0), ('doc6', 2, 14.0)]
    }

    run2 = {
        'q1': [('doc2', 1, 20.0), ('doc1', 2, 19.0), ('doc4', 3, 18.0)],
        'q2': [('doc6', 1, 25.0), ('doc5', 2, 24.0)]
    }

    # Fuse with RRF
    fused = reciprocal_rank_fusion([run1, run2], k=60)

    # Verify structure
    assert 'q1' in fused
    assert 'q2' in fused

    # doc1 and doc2 should be top results for q1 (both appear in both runs)
    q1_docs = [docid for docid, _ in fused['q1']]
    assert 'doc1' in q1_docs[:2]
    assert 'doc2' in q1_docs[:2]


def test_linear_combination():
    """Test linear combination fusion."""
    # Create mock runs
    run1 = {
        'q1': [('doc1', 1, 10.0), ('doc2', 2, 5.0), ('doc3', 3, 1.0)],
    }

    run2 = {
        'q1': [('doc2', 1, 100.0), ('doc1', 2, 50.0), ('doc4', 3, 10.0)],
    }

    # Equal weights
    fused = linear_combination([run1, run2], weights=[0.5, 0.5])

    assert 'q1' in fused
    q1_results = fused['q1']

    # Both doc1 and doc2 should appear
    q1_docs = [docid for docid, _ in q1_results]
    assert 'doc1' in q1_docs
    assert 'doc2' in q1_docs


def test_weighted_combination():
    """Test weighted linear combination."""
    run1 = {
        'q1': [('doc1', 1, 100.0), ('doc2', 2, 50.0)],
    }

    run2 = {
        'q1': [('doc2', 1, 100.0), ('doc3', 2, 50.0)],
    }

    # Heavy weight on run1
    fused = linear_combination([run1, run2], weights=[0.9, 0.1])

    q1_docs = [docid for docid, _ in fused['q1']]

    # doc1 should benefit from run1's high weight
    assert q1_docs[0] in ['doc1', 'doc2']


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
