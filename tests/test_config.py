"""Tests for config module."""

import pytest
import config


def test_length_boundaries():
    """Verify boundaries 1, 5, 6, 10, 11 map to the correct length labels."""
    assert config.length_from_line_count(1) == "Short"
    assert config.length_from_line_count(5) == "Short"
    assert config.length_from_line_count(6) == "Medium"
    assert config.length_from_line_count(10) == "Medium"
    assert config.length_from_line_count(11) == "Long"
    assert config.length_from_line_count(25) == "Long"


def test_length_to_text():
    """Verify length_to_text returns agreed line ranges."""
    assert config.length_to_text("Short") == "1 to 5 lines"
    assert config.length_to_text("Medium") == "6 to 10 lines"
    assert config.length_to_text("Long") == "11 to 15 lines"


def test_constants():
    """Verify core constants and limits."""
    assert "English" in config.LANGUAGES
    assert "Hinglish" in config.LANGUAGES
    assert config.MAX_EXAMPLES == 2
    assert config.MAX_TAGS_PER_POST == 2
