"""Configuration constants and utility functions for LinkedIn Post Generator."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw_posts.json"
PROCESSED_DATA_PATH = DATA_DIR / "processed_posts.json"

# Length definitions
# Maps label to (min_lines, max_lines) tuple (inclusive)
LENGTH_RANGES: dict[str, tuple[int, int]] = {
    "Short": (1, 5),
    "Medium": (6, 10),
    "Long": (11, 999),
}

# Human-readable line ranges for prompt and UI consistency
LENGTH_TEXT_MAPPINGS: dict[str, str] = {
    "Short": "1 to 5 lines",
    "Medium": "6 to 10 lines",
    "Long": "11 to 15 lines",
}

# Supported languages
LANGUAGES: list[str] = ["English", "Hinglish"]

# Limits
MAX_EXAMPLES: int = 2
MAX_TAGS_PER_POST: int = 2

# Model defaults
DEFAULT_MODEL: str = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE: float = 0.7


def length_from_line_count(n: int) -> str:
    """Map a non-empty line count to a length category ("Short", "Medium", "Long")."""
    for label, (min_lines, max_lines) in LENGTH_RANGES.items():
        if min_lines <= n <= max_lines:
            return label
    return "Long" if n > 10 else "Short"


def length_to_text(label: str) -> str:
    """Return explicit line range text for a given length category."""
    return LENGTH_TEXT_MAPPINGS.get(label, "6 to 10 lines")
