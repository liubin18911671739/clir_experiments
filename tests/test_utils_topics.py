"""Tests for utils_topics module."""

import tempfile
from pathlib import Path
import pytest
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils_topics import parse_trec_topics, load_qrels, write_trec_topics


def test_parse_xml_style_topics():
    """Test parsing XML-style TREC topics."""
    topics_content = """
<top>
<num> Number: 1
<title> cross-lingual information retrieval
<desc> Description:
Find documents about cross-lingual IR systems.
<narr> Narrative:
Relevant documents discuss CLIR methods.
</top>

<top>
<num> Number: 2
<title> neural reranking models
</top>
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        topics_path = Path(tmpdir) / "topics.txt"

        with open(topics_path, 'w', encoding='utf-8') as f:
            f.write(topics_content)

        # Parse without desc/narr
        queries = parse_trec_topics(str(topics_path))
        assert len(queries) == 2
        assert '1' in queries
        assert '2' in queries
        assert queries['1'] == 'cross-lingual information retrieval'
        assert queries['2'] == 'neural reranking models'

        # Parse with desc
        queries_with_desc = parse_trec_topics(str(topics_path), use_desc=True)
        assert 'Find documents' in queries_with_desc['1']


def test_parse_simple_topics():
    """Test parsing simple topic format."""
    topics_content = """1\tcross-lingual information retrieval
2\tneural reranking models
3\tdense retrieval methods
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        topics_path = Path(tmpdir) / "topics.txt"

        with open(topics_path, 'w', encoding='utf-8') as f:
            f.write(topics_content)

        queries = parse_trec_topics(str(topics_path))
        assert len(queries) == 3
        assert queries['1'] == 'cross-lingual information retrieval'
        assert queries['2'] == 'neural reranking models'
        assert queries['3'] == 'dense retrieval methods'


def test_write_read_topics():
    """Test writing and reading topics."""
    test_topics = {
        '1': 'cross-lingual information retrieval',
        '2': 'neural reranking',
        '3': 'dense retrieval'
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        topics_path = Path(tmpdir) / "topics.txt"

        # Write simple format
        write_trec_topics(test_topics, str(topics_path), format='simple')
        assert topics_path.exists()

        # Read back
        loaded_topics = parse_trec_topics(str(topics_path))
        assert loaded_topics == test_topics

        # Write XML format
        topics_path_xml = Path(tmpdir) / "topics_xml.txt"
        write_trec_topics(test_topics, str(topics_path_xml), format='xml')
        loaded_topics_xml = parse_trec_topics(str(topics_path_xml))
        assert loaded_topics_xml == test_topics


def test_load_qrels():
    """Test loading qrels file."""
    qrels_content = """1 0 doc1 1
1 0 doc2 2
1 0 doc3 0
2 0 doc5 1
2 0 doc1 1
3 0 doc7 2
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        qrels_path = Path(tmpdir) / "qrels.txt"

        with open(qrels_path, 'w', encoding='utf-8') as f:
            f.write(qrels_content)

        qrels = load_qrels(str(qrels_path))

        # Verify structure
        assert '1' in qrels
        assert '2' in qrels
        assert '3' in qrels

        # Verify contents
        assert qrels['1']['doc1'] == 1
        assert qrels['1']['doc2'] == 2
        assert qrels['1']['doc3'] == 0
        assert qrels['2']['doc5'] == 1
        assert qrels['3']['doc7'] == 2


def test_empty_topics():
    """Test handling of empty topics file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        topics_path = Path(tmpdir) / "empty_topics.txt"

        with open(topics_path, 'w', encoding='utf-8') as f:
            f.write("")

        queries = parse_trec_topics(str(topics_path))
        assert len(queries) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
