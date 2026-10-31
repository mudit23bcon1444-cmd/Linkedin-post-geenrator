"""Tests for preprocess module (no network, mocked LLM)."""

from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage
from preprocess import (
    count_non_empty_lines,
    extract_json_from_text,
    analyze_single_post,
    unify_tags,
)


def test_count_non_empty_lines():
    """Verify line_count is computed in Python ignoring empty lines."""
    text_1 = "Line 1\nLine 2\n\nLine 3\n  \nLine 4"
    assert count_non_empty_lines(text_1) == 4

    text_2 = "\n\n   \n"
    assert count_non_empty_lines(text_2) == 0


def test_extract_json_from_text_fenced():
    """Verify fenced JSON with markdown blocks and extra prose parses correctly."""
    fenced_input = """```json
{
  "language": "English",
  "tags": ["DSA Practice", "Interviews"]
}
```"""
    data = extract_json_from_text(fenced_input)
    assert data["language"] == "English"
    assert data["tags"] == ["DSA Practice", "Interviews"]

    prose_input = """Here is the extracted metadata:
{
  "language": "Hinglish",
  "tags": ["Projects"]
}
Hope this helps!"""
    data_prose = extract_json_from_text(prose_input)
    assert data_prose["language"] == "Hinglish"
    assert data_prose["tags"] == ["Projects"]


def test_invalid_language_defaults_to_english():
    """Verify invalid language value defaults to English."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = Exception("Not supported")
    
    mock_resp = AIMessage(content='{"language": "Spanish", "tags": ["DSA Practice"]}')
    mock_llm.invoke.return_value = mock_resp
    mock_llm.return_value = mock_resp

    result = analyze_single_post("Some post text", mock_llm)
    assert result is not None
    assert result["language"] == "English"
    assert result["tags"] == ["DSA Practice"]


def test_more_than_two_tags_are_trimmed():
    """Verify more than 2 tags are trimmed to config.MAX_TAGS_PER_POST (2)."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = Exception("Not supported")

    mock_resp = AIMessage(content='{"language": "English", "tags": ["DSA", "LeetCode", "Interviews", "Coding"]}')
    mock_llm.invoke.return_value = mock_resp
    mock_llm.return_value = mock_resp

    result = analyze_single_post("Some post text", mock_llm)
    assert result is not None
    assert len(result["tags"]) == 2
    assert result["tags"] == ["DSA", "LeetCode"]


def test_bad_first_response_triggers_retry():
    """Verify a malformed first response triggers a retry and succeeds on subsequent attempt."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = Exception("Not supported")

    bad_resp = AIMessage(content="Malformed not-json response")
    good_resp = AIMessage(content='{"language": "Hinglish", "tags": ["Internships"]}')

    mock_llm.invoke.side_effect = [bad_resp, good_resp]
    mock_llm.side_effect = [bad_resp, good_resp]

    result = analyze_single_post("Post text", mock_llm, max_retries=3)
    assert result is not None
    assert result["language"] == "Hinglish"
    assert result["tags"] == ["Internships"]


def test_unify_tags():
    """Verify unify_tags parses mapping and formats Title Case."""
    mock_llm = MagicMock()
    mock_resp = AIMessage(content='{"dsa": "DSA Practice", "job hunt": "Placements"}')
    mock_llm.invoke.return_value = mock_resp
    mock_llm.return_value = mock_resp

    mapping = unify_tags(["dsa", "job hunt"], mock_llm)
    assert mapping["dsa"] == "DSA Practice"
    assert mapping["job hunt"] == "Placements"
