"""
Utility functions for parsing TREC/NeuCLIR topic files.

Supports standard TREC topic formats with <num>, <title>, <desc>, and <narr> fields.
"""

import re
from pathlib import Path
from typing import Dict, List


class Topic:
    """Represents a single TREC topic/query."""

    def __init__(self, num: str, title: str, desc: str = "", narr: str = ""):
        """
        Initialize a Topic.

        Args:
            num: Topic number/ID
            title: Topic title (short query)
            desc: Topic description (optional)
            narr: Topic narrative (optional)
        """
        self.num = num
        self.title = title.strip()
        self.desc = desc.strip()
        self.narr = narr.strip()

    def __repr__(self) -> str:
        return f"Topic(num={self.num}, title='{self.title[:50]}...')"

    def get_query_text(self, use_desc: bool = False, use_narr: bool = False) -> str:
        """
        Get query text for this topic.

        Args:
            use_desc: Include description in query text
            use_narr: Include narrative in query text

        Returns:
            Query text string
        """
        parts = [self.title]

        if use_desc and self.desc:
            parts.append(self.desc)

        if use_narr and self.narr:
            parts.append(self.narr)

        return " ".join(parts)


def parse_trec_topics(topics_path: str, use_desc: bool = False, use_narr: bool = False) -> Dict[str, str]:
    """
    Parse TREC-format topics file into query dictionary.

    Supports two common formats:
    1. XML-style with <top>, <num>, <title>, <desc>, <narr> tags
    2. Simple format with just topic numbers and titles

    Args:
        topics_path: Path to TREC topics file
        use_desc: Include description field in query text
        use_narr: Include narrative field in query text

    Returns:
        Dictionary mapping topic_id to query_text

    Raises:
        FileNotFoundError: If topics file doesn't exist
    """
    topics_path = Path(topics_path)
    if not topics_path.exists():
        raise FileNotFoundError(f"Topics file not found: {topics_path}")

    with open(topics_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Try XML-style format first
    topics = _parse_xml_style_topics(content)

    # If no topics found, try simple format
    if not topics:
        topics = _parse_simple_topics(content)

    # Convert to query dictionary
    queries = {}
    for topic in topics:
        queries[topic.num] = topic.get_query_text(use_desc=use_desc, use_narr=use_narr)

    return queries


def _parse_xml_style_topics(content: str) -> List[Topic]:
    """
    Parse XML-style TREC topics.

    Expected format:
    <top>
    <num> Number: 301
    <title> Topic title here
    <desc> Description:
    Topic description here
    <narr> Narrative:
    Topic narrative here
    </top>

    Args:
        content: Raw topic file content

    Returns:
        List of Topic objects
    """
    topics = []

    # Split into individual topics
    topic_pattern = re.compile(r'<top>(.*?)</top>', re.DOTALL | re.IGNORECASE)
    topic_matches = topic_pattern.findall(content)

    for topic_text in topic_matches:
        # Extract fields
        num = _extract_field(topic_text, r'<num>(?:\s*Number:)?\s*(\S+)')
        title = _extract_field(topic_text, r'<title>\s*(.*?)(?=<|$)', multiline=True)
        desc = _extract_field(topic_text, r'<desc>(?:\s*Description:)?\s*(.*?)(?=<|$)', multiline=True)
        narr = _extract_field(topic_text, r'<narr>(?:\s*Narrative:)?\s*(.*?)(?=<|$)', multiline=True)

        if num:
            topics.append(Topic(num=num, title=title, desc=desc, narr=narr))

    return topics


def _parse_simple_topics(content: str) -> List[Topic]:
    """
    Parse simple topic format (one topic per line).

    Expected format:
    301 topic title here
    302 another topic title

    Or:
    301\ttopic title here
    302\tanother topic title

    Args:
        content: Raw topic file content

    Returns:
        List of Topic objects
    """
    topics = []

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try to split on whitespace or tab
        parts = re.split(r'\s+', line, maxsplit=1)
        if len(parts) >= 2:
            num = parts[0]
            title = parts[1]
            topics.append(Topic(num=num, title=title))
        # Also try tab-separated
        elif '\t' in line:
            parts = line.split('\t', maxsplit=1)
            if len(parts) >= 2:
                num = parts[0].strip()
                title = parts[1].strip()
                topics.append(Topic(num=num, title=title))

    return topics


def _extract_field(text: str, pattern: str, multiline: bool = False) -> str:
    """
    Extract a field from topic text using regex.

    Args:
        text: Topic text to search
        pattern: Regex pattern with one capture group
        multiline: Whether to use multiline matching

    Returns:
        Extracted field text, or empty string if not found
    """
    flags = re.IGNORECASE
    if multiline:
        flags |= re.DOTALL

    match = re.search(pattern, text, flags)
    if match:
        return match.group(1).strip()
    return ""


def write_trec_topics(topics: Dict[str, str], output_path: str, format: str = "simple") -> None:
    """
    Write topics to TREC-format file.

    Args:
        topics: Dictionary mapping topic_id to query_text
        output_path: Path to output topics file
        format: Output format ('simple' or 'xml')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        if format == "simple":
            for num, query in sorted(topics.items()):
                f.write(f"{num}\t{query}\n")
        elif format == "xml":
            for num, query in sorted(topics.items()):
                f.write("<top>\n")
                f.write(f"<num> Number: {num}\n")
                f.write(f"<title> {query}\n")
                f.write("</top>\n\n")
        else:
            raise ValueError(f"Unknown format: {format}")


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load TREC-format qrels file.

    Expected format: qid iter docid relevance

    Args:
        qrels_path: Path to qrels file

    Returns:
        Dictionary mapping qid -> {docid: relevance_score}

    Raises:
        FileNotFoundError: If qrels file doesn't exist
    """
    qrels_path = Path(qrels_path)
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    qrels: Dict[str, Dict[str, int]] = {}

    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                docid = parts[2]
                relevance = int(parts[3])

                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = relevance

    return qrels
