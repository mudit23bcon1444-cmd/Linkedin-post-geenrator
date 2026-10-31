"""Stage 1: Preprocess and enrich raw posts with metadata using LLM."""

import json
import re
import sys
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
import config
from llm import get_llm


class PostMetadata(BaseModel):
    """Pydantic schema for post enrichment."""
    language: str = Field(
        description="Language of the post: 'English' or 'Hinglish'",
        default="English"
    )
    tags: list[str] = Field(
        description="Up to 2 concise topic tags (e.g. ['DSA Practice', 'Resume Tips'])",
        default_factory=list
    )


TAG_EXTRACTION_PROMPT = """You are analyzing a LinkedIn post written by a computer science student preparing for placements.
Analyze the following post text and identify:
1. Language: Either "English" or "Hinglish" (Hinglish is a mixture of Hindi and English written in Roman/Latin script).
2. Up to 2 relevant topic tags (e.g., DSA Practice, Resume Tips, Placements, Internships, Projects, Mental Health, Motivation).

Post text:
\"\"\"{post_text}\"\"\"

Respond with a valid JSON object matching this schema:
{{
  "language": "English" or "Hinglish",
  "tags": ["Tag 1", "Tag 2"]
}}
Return ONLY the JSON object. Do not include markdown formatting or extra text."""


TAG_UNIFICATION_PROMPT = """You are a taxonomy expert standardizing topic tags for LinkedIn posts.
Below is a list of raw topic tags extracted from posts:
{raw_tags}

Unify and standardize these tags into clean, Title Case categories, merging near-duplicates and synonyms.
Every input tag MUST be mapped to a standardized tag.

Few-shot examples:
1. "dsa" -> "DSA Practice"
2. "Job Search" -> "Placements"
3. "Interview Preparation" -> "Placements"
4. "Resume Building" -> "Resume Tips"

Return ONLY a valid JSON object mapping each raw tag to its unified Title Case tag, like:
{{
  "raw tag 1": "Unified Tag 1",
  "raw tag 2": "Unified Tag 2"
}}
Return ONLY the JSON object. Do not include explanation or markdown code fences."""


def count_non_empty_lines(text: str) -> int:
    """Compute line count in Python by counting non-empty lines."""
    return len([line for line in text.split("\n") if line.strip()])


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract and parse JSON from text, stripping markdown code fences and surrounding prose."""
    cleaned = text.strip()
    # Strip markdown code blocks ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # Find outermost curly braces if extra prose exists
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    return json.loads(cleaned)


def analyze_single_post(
    post_text: str,
    llm: Any,
    max_retries: int = 3,
) -> Optional[dict[str, Any]]:
    """Enrich a single post with language and up to 2 tags using LLM with retries."""
    prompt = PromptTemplate.from_template(TAG_EXTRACTION_PROMPT)

    # Attempt structured output first if available
    structured_llm = None
    try:
        structured_llm = llm.with_structured_output(PostMetadata)
    except Exception:
        structured_llm = None

    for attempt in range(1, max_retries + 1):
        try:
            language = None
            tags = None

            if structured_llm is not None:
                try:
                    structured_chain = prompt | structured_llm
                    result = structured_chain.invoke({"post_text": post_text})
                    if isinstance(result, PostMetadata):
                        language = result.language
                        tags = result.tags
                    elif isinstance(result, dict):
                        language = result.get("language")
                        tags = result.get("tags")
                except Exception:
                    # Fallback to direct prompt | llm and text extraction
                    pass

            if language is None or tags is None:
                text_chain = prompt | llm
                response = text_chain.invoke({"post_text": post_text})
                if isinstance(getattr(response, "content", None), str):
                    content = response.content
                elif isinstance(response, str):
                    content = response
                else:
                    content = str(response)
                parsed = extract_json_from_text(content)
                language = parsed.get("language")
                tags = parsed.get("tags")

            # Validate language
            if not isinstance(language, str) or language not in config.LANGUAGES:
                language = "English"

            # Validate tags: list of up to 2 non-empty strings
            if not isinstance(tags, list):
                tags = []
            cleaned_tags = [str(t).strip() for t in tags if str(t).strip()]
            cleaned_tags = cleaned_tags[:config.MAX_TAGS_PER_POST]

            return {
                "language": language,
                "tags": cleaned_tags,
            }
        except Exception as e:
            if attempt == max_retries:
                print(f"Warning: Failed to analyze post after {max_retries} attempts: {e}", file=sys.stderr)
                return None

    return None


def unify_tags(unique_tags: list[str], llm: Any) -> dict[str, str]:
    """Unify unique tags into standardized Title Case categories using LLM call 2."""
    if not unique_tags:
        return {}

    prompt = PromptTemplate.from_template(TAG_UNIFICATION_PROMPT)
    chain = prompt | llm

    try:
        response = chain.invoke({"raw_tags": json.dumps(unique_tags, indent=2)})
        if isinstance(getattr(response, "content", None), str):
            content = response.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)
        mapping = extract_json_from_text(content)
        if isinstance(mapping, dict):
            return {
                str(k).strip(): str(v).strip()
                for k, v in mapping.items()
                if str(k).strip() and str(v).strip()
            }
    except Exception as e:
        print(f"Warning: Tag unification LLM call failed ({e}). Using raw tags.")

    return {tag: tag.strip() for tag in unique_tags}


def preprocess_data(
    raw_path: str = str(config.RAW_DATA_PATH),
    processed_path: str = str(config.PROCESSED_DATA_PATH),
    llm: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Run Stage 1 preprocessing on raw posts dataset."""
    if llm is None:
        llm = get_llm()

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_posts = json.load(f)

    print(f"Loaded {len(raw_posts)} raw posts from {raw_path}")

    processed_posts: list[dict[str, Any]] = []
    skipped_indices: list[int] = []

    # Step 1: Compute line counts in Python and enrich each post
    for idx, post in enumerate(raw_posts):
        text = post.get("text", "")
        engagement = post.get("engagement", 100)
        line_count = count_non_empty_lines(text)

        metadata = analyze_single_post(text, llm)
        if metadata is None:
            print(f"Skipping post index {idx} due to metadata extraction failure.")
            skipped_indices.append(idx)
            continue

        processed_posts.append({
            "text": text,
            "engagement": engagement,
            "line_count": line_count,
            "language": metadata["language"],
            "tags": metadata["tags"],
        })
        print(f"Processed post {idx + 1}/{len(raw_posts)}: lines={line_count}, lang={metadata['language']}, tags={metadata['tags']}")

    # Step 2: Unify tags across the whole dataset
    all_raw_tags: set[str] = set()
    for p in processed_posts:
        for t in p["tags"]:
            all_raw_tags.add(t)

    sorted_raw_tags = sorted(list(all_raw_tags))
    print(f"\nExtracted {len(sorted_raw_tags)} unique raw tags. Unifying tags...")
    tag_mapping = unify_tags(sorted_raw_tags, llm)

    # Step 3: Apply unified tags and deduplicate per post
    for p in processed_posts:
        unified_tags = []
        for t in p["tags"]:
            mapped = tag_mapping.get(t, t.strip())
            if mapped and mapped not in unified_tags:
                unified_tags.append(mapped)
        p["tags"] = unified_tags[:config.MAX_TAGS_PER_POST]

    # Save to processed JSON
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(processed_posts, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully wrote {len(processed_posts)} records to {processed_path}")
    if skipped_indices:
        print(f"Skipped indices: {skipped_indices}")

    # Summary
    all_unified_tags = sorted(list({t for p in processed_posts for t in p["tags"]}))
    print(f"Final Unified Tags ({len(all_unified_tags)}): {all_unified_tags}")

    return processed_posts


if __name__ == "__main__":
    preprocess_data()
