"""Stage 2b: Prompt construction and post generation module."""

import re
from typing import Any, Optional
import config
from llm import get_llm
from retriever import PostRetriever


def build_prompt(
    length: str,
    language: str,
    tag: str,
    examples: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Construct the few-shot prompt for LinkedIn post generation.

    Args:
        length: Selected length label ("Short", "Medium", "Long")
        language: Target language ("English" or "Hinglish")
        tag: Chosen topic tag
        examples: Optional list of retrieved example post records

    Returns:
        Formatted prompt string.
    """
    line_range_text = config.length_to_text(length)

    prompt_lines = [
        "Generate a LinkedIn post. No preamble.",
        f"Topic: {tag}",
        f"Length: {line_range_text} (strictly adhere to this line count, where each line is non-empty separated by a newline)",
    ]

    if language.lower() == "hinglish":
        prompt_lines.append(
            "Language: Hinglish (a natural, conversational blend of Hindi and English words). "
            "CRITICAL: The script must ALWAYS be written exclusively in English/Roman Latin alphabet. "
            "Never use Devanagari script."
        )
    else:
        prompt_lines.append(f"Language: {language}")

    # Voice and formatting guidance
    prompt_lines.append(
        "Tone and Voice: Short punchy lines, strong hook in the first line, personal storytelling tone, "
        "at most 1-2 relevant emojis, and end with a clear one-line takeaway or question."
    )

    if examples:
        prompt_lines.append(
            "\nUse the writing style of these examples (structure, cadence, sentence length, and vocabulary). "
            "Do NOT copy the examples verbatim:"
        )
        for i, ex in enumerate(examples, 1):
            ex_text = ex.get("text", "") if isinstance(ex, dict) else str(ex)
            prompt_lines.append(f"\n--- Example {i} ---\n{ex_text.strip()}")

    prompt_lines.append("\nWrite the final LinkedIn post below with NO opening preamble, commentary, or markdown fences:")
    return "\n".join(prompt_lines)


def strip_preamble(text: str) -> str:
    """Strip conversational preambles (e.g., 'Here is your post:') and surrounding fences."""
    cleaned = text.strip()

    # Strip markdown code blocks if wrapped
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # Regex patterns for typical LLM preambles
    preamble_patterns = [
        r"^(?:Here\s+(?:is|are)|Sure,?\s*(?:here\s+is)?|Certainly!?\s*(?:here\s+is)?).*?:\s*\n*",
        r"^(?:LinkedIn Post|Draft Post|Generated Post):\s*\n*",
        r"^Here'?s\s+(?:a|your|the)\s+LinkedIn\s+post.*?:?\s*\n*",
    ]

    for pattern in preamble_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def generate_post(
    length: str,
    language: str,
    tag: str,
    retriever: Optional[PostRetriever] = None,
    examples: Optional[list[dict[str, Any]]] = None,
    match_level: Optional[str] = None,
    llm: Optional[Any] = None,
    max_retries: int = 2,
) -> tuple[str, list[dict[str, Any]], str]:
    """Generate a LinkedIn post using retrieved examples and Groq LLM.

    Args:
        length: Length category ("Short", "Medium", "Long")
        language: Language ("English", "Hinglish")
        tag: Topic tag
        retriever: Optional PostRetriever instance
        examples: Optional explicitly provided example posts
        match_level: Optional match level string
        llm: Optional LLM instance
        max_retries: Number of retries on failure (default 2)

    Returns:
        Tuple of (generated_post_text, examples_used, match_level)
    """
    if examples is None or match_level is None:
        if retriever is not None:
            examples, match_level = retriever.get_examples(length, language, tag)
        else:
            examples, match_level = [], "none"

    if llm is None:
        llm = get_llm()

    prompt_str = build_prompt(length, language, tag, examples)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        try:
            response = llm.invoke(prompt_str)
            raw_text = response.content if hasattr(response, "content") else str(response)
            clean_text = strip_preamble(raw_text)
            if clean_text:
                return clean_text, examples, match_level
        except Exception as e:
            last_error = e
            if attempt > max_retries:
                raise RuntimeError(f"Failed to generate post after {max_retries} retries: {last_error}") from last_error

    raise RuntimeError(f"Failed to generate post: {last_error}")
