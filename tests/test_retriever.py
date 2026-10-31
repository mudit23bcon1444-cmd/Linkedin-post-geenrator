"""Tests for retriever module (no network)."""

import json
import pytest
from retriever import PostRetriever


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a temporary processed JSON dataset for deterministic tests."""
    posts = [
        {
            "text": "Post 1: DSA in English Short high engagement",
            "engagement": 800,
            "line_count": 4,
            "language": "English",
            "tags": ["DSA Practice", "Interviews"],
        },
        {
            "text": "Post 2: DSA in English Short lower engagement",
            "engagement": 300,
            "line_count": 3,
            "language": "English",
            "tags": ["DSA Practice"],
        },
        {
            "text": "Post 3: DSA in English Medium",
            "engagement": 600,
            "line_count": 8,
            "language": "English",
            "tags": ["DSA Practice"],
        },
        {
            "text": "Post 4: DSA in Hinglish Short",
            "engagement": 750,
            "line_count": 5,
            "language": "Hinglish",
            "tags": ["DSA Practice"],
        },
        {
            "text": "Post 5: Resume in Hinglish Long",
            "engagement": 500,
            "line_count": 12,
            "language": "Hinglish",
            "tags": ["Resume Tips"],
        },
    ]
    file_path = tmp_path / "processed_posts.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(posts, f)
    return file_path


def test_get_tags_sorted_and_unique(sample_dataset):
    """Verify retriever.get_tags returns unique and sorted list of tags."""
    retriever = PostRetriever(sample_dataset)
    tags = retriever.get_tags()
    assert tags == ["DSA Practice", "Interviews", "Resume Tips"]


def test_exact_match_level_1(sample_dataset):
    """Verify level 1 exact match (tag + language + length) and engagement sorting."""
    retriever = PostRetriever(sample_dataset)
    examples, match_level = retriever.get_examples(
        length="Short",
        language="English",
        tag="DSA Practice",
        k=2,
    )
    assert match_level == "tag + language + length"
    assert len(examples) == 2
    # Verify sorted by engagement descending: 800 then 300
    assert examples[0]["engagement"] == 800
    assert examples[1]["engagement"] == 300


def test_fallback_level_2_tag_and_language(sample_dataset):
    """Verify fallback to level 2 (tag + language) when length doesn't match."""
    retriever = PostRetriever(sample_dataset)
    # No Long posts for DSA Practice in English (only Short and Medium exist)
    examples, match_level = retriever.get_examples(
        length="Long",
        language="English",
        tag="DSA Practice",
        k=2,
    )
    assert match_level == "tag + language"
    assert len(examples) == 2
    # Should get highest engagement English posts for DSA Practice: 800 and 600
    assert examples[0]["engagement"] == 800
    assert examples[1]["engagement"] == 600


def test_fallback_level_3_tag_only(sample_dataset):
    """Verify fallback to level 3 (tag only) when language doesn't match."""
    retriever = PostRetriever(sample_dataset)
    # Resume Tips only exists in Hinglish, request English
    examples, match_level = retriever.get_examples(
        length="Short",
        language="English",
        tag="Resume Tips",
        k=2,
    )
    assert match_level == "tag only"
    assert len(examples) == 1
    assert examples[0]["tags"] == ["Resume Tips"]


def test_fallback_level_4_none(sample_dataset):
    """Verify level 4 none when tag does not exist at all."""
    retriever = PostRetriever(sample_dataset)
    examples, match_level = retriever.get_examples(
        length="Short",
        language="English",
        tag="NonExistentTag",
        k=2,
    )
    assert match_level == "none"
    assert examples == []
