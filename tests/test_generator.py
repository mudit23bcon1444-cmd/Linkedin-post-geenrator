"""Tests for generator module (no network, mocked LLM)."""

from unittest.mock import MagicMock
import pytest
from generator import build_prompt, strip_preamble, generate_post


def test_build_prompt_contains_required_fields():
    """Verify prompt includes topic, line range, and language details."""
    prompt = build_prompt(
        length="Short",
        language="English",
        tag="DSA Practice",
        examples=None,
    )
    assert "Topic: DSA Practice" in prompt
    assert "1 to 5 lines" in prompt
    assert "Language: English" in prompt
    assert "Generate a LinkedIn post. No preamble." in prompt
    assert "Example 1" not in prompt


def test_build_prompt_hinglish_instruction():
    """Verify Hinglish explains Roman/Latin script and Hindi+English mix."""
    prompt = build_prompt(
        length="Medium",
        language="Hinglish",
        tag="Resume Tips",
        examples=None,
    )
    assert "Language: Hinglish" in prompt
    assert "Roman Latin alphabet" in prompt
    assert "Never use Devanagari" in prompt
    assert "6 to 10 lines" in prompt


def test_build_prompt_with_examples():
    """Verify 'Example 1' appears only when examples are provided."""
    examples = [
        {"text": "First sample post text.", "engagement": 500},
        {"text": "Second sample post text.", "engagement": 400},
    ]
    prompt = build_prompt(
        length="Long",
        language="English",
        tag="Projects",
        examples=examples,
    )
    assert "Example 1" in prompt
    assert "Example 2" in prompt
    assert "First sample post text." in prompt
    assert "Second sample post text." in prompt
    assert "Use the writing style of these examples" in prompt
    assert "11 to 15 lines" in prompt


def test_strip_preamble_various_formats():
    """Verify preamble stripping removes typical chat prefixes and code fences."""
    raw_1 = "Here is your post:\n\nSolved 100 questions today.\nConsistency is key!"
    assert strip_preamble(raw_1) == "Solved 100 questions today.\nConsistency is key!"

    raw_2 = "Sure! Here is a LinkedIn post for you:\n\nDSA is simple once you practice."
    assert strip_preamble(raw_2) == "DSA is simple once you practice."

    raw_3 = "```markdown\nHere's the LinkedIn post:\nNo shortcuts in engineering.\n```"
    assert strip_preamble(raw_3) == "No shortcuts in engineering."

    clean_text = "Mastering graph algorithms in college.\nBFS and DFS are fundamental."
    assert strip_preamble(clean_text) == clean_text


def test_generate_post_with_mock_llm():
    """Verify generate_post invokes LLM and strips preamble correctly."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Here is your post:\nFirst hook line.\nBody line.\nTakeaway line."
    mock_llm.invoke.return_value = mock_response

    examples = [{"text": "Sample text", "engagement": 500}]
    result_text, used_examples, match_level = generate_post(
        length="Short",
        language="English",
        tag="DSA Practice",
        examples=examples,
        match_level="tag + language + length",
        llm=mock_llm,
    )

    assert mock_llm.invoke.called
    assert result_text == "First hook line.\nBody line.\nTakeaway line."
    assert used_examples == examples
    assert match_level == "tag + language + length"
