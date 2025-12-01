"""Tests for utils_io module."""

import json
import tempfile
from pathlib import Path
import pytest
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils_io import (
    load_yaml, save_yaml, ensure_dir, write_trec_run,
    read_trec_run, load_jsonl
)


def test_ensure_dir():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "subdir" / "nested"
        result = ensure_dir(test_dir)

        assert result.exists()
        assert result.is_dir()


def test_load_save_yaml():
    """Test YAML loading and saving."""
    test_data = {
        'key1': 'value1',
        'key2': [1, 2, 3],
        'key3': {'nested': 'data'}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "test.yaml"

        # Save
        save_yaml(test_data, str(yaml_path))
        assert yaml_path.exists()

        # Load
        loaded_data = load_yaml(str(yaml_path))
        assert loaded_data == test_data


def test_write_read_trec_run():
    """Test TREC run file writing and reading."""
    test_results = [
        ('q1', 'doc1', 10.5),
        ('q1', 'doc2', 9.2),
        ('q1', 'doc3', 8.1),
        ('q2', 'doc5', 15.3),
        ('q2', 'doc1', 12.0),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        run_path = Path(tmpdir) / "test.run"

        # Write
        write_trec_run(test_results, str(run_path), "test_run")
        assert run_path.exists()

        # Read
        run_data = read_trec_run(str(run_path))

        # Verify structure
        assert 'q1' in run_data
        assert 'q2' in run_data
        assert len(run_data['q1']) == 3
        assert len(run_data['q2']) == 2

        # Verify sorting by rank
        q1_docs = [docid for docid, _, _ in run_data['q1']]
        assert q1_docs == ['doc1', 'doc2', 'doc3']


def test_load_jsonl():
    """Test JSONL loading."""
    test_docs = [
        {'id': 'doc1', 'contents': 'First document'},
        {'id': 'doc2', 'contents': 'Second document'},
        {'id': 'doc3', 'contents': 'Third document'},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "corpus.jsonl"

        # Write test data
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for doc in test_docs:
                f.write(json.dumps(doc) + '\n')

        # Load and verify
        loaded_docs = list(load_jsonl(str(jsonl_path)))
        assert len(loaded_docs) == 3
        assert loaded_docs == test_docs


def test_trec_run_format():
    """Test TREC run file format compliance."""
    test_results = [
        ('q1', 'doc1', 10.5),
        ('q1', 'doc2', 9.2),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        run_path = Path(tmpdir) / "test.run"
        write_trec_run(test_results, str(run_path), "test_run")

        # Read raw file and check format
        with open(run_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Check first line format: qid Q0 docid rank score runid
        parts = lines[0].strip().split()
        assert len(parts) == 6
        assert parts[0] == 'q1'
        assert parts[1] == 'Q0'
        assert parts[2] == 'doc1'
        assert parts[3] == '1'  # rank
        assert parts[5] == 'test_run'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
