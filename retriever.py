"""Stage 2a: PostRetriever module for loading, filtering, and falling back on post examples."""

from pathlib import Path
from typing import Any, Optional, Union
import json
import pandas as pd
import config


class PostRetriever:
    """Retrieves few-shot LinkedIn post examples based on user selection with fallback."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize retriever by loading processed posts into pandas DataFrame."""
        self.data_path = Path(data_path or config.PROCESSED_DATA_PATH)
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare DataFrame with derived length column."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Processed data file not found at: {self.data_path}. "
                "Please run `python preprocess.py` first."
            )

        with open(self.data_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            return pd.DataFrame(columns=["text", "engagement", "line_count", "language", "tags", "length"])

        df = pd.DataFrame(records)
        # Compute derived length category
        df["length"] = df["line_count"].apply(config.length_from_line_count)
        return df

    def get_tags(self) -> list[str]:
        """Return a sorted unique list of all topic tags present in the dataset."""
        if self.df.empty:
            return []

        all_tags = set()
        for tag_list in self.df["tags"]:
            if isinstance(tag_list, list):
                for t in tag_list:
                    if t and str(t).strip():
                        all_tags.add(str(t).strip())

        return sorted(list(all_tags))

    def get_examples(
        self,
        length: str,
        language: str,
        tag: str,
        k: int = config.MAX_EXAMPLES,
    ) -> tuple[list[dict[str, Any]], str]:
        """Retrieve up to k posts matching criteria, following a 4-tier fallback chain.

        Fallback tiers:
        1. tag + language + length
        2. tag + language
        3. tag only
        4. none

        Args:
            length: Target length label ("Short", "Medium", "Long")
            language: Target language ("English", "Hinglish")
            tag: Target topic tag (e.g. "DSA Practice")
            k: Maximum number of examples to return (sorted by engagement desc)

        Returns:
            Tuple of (list_of_example_records, match_level_string)
        """
        if self.df.empty:
            return [], "none"

        has_tag = self.df["tags"].apply(lambda tags: isinstance(tags, list) and tag in tags)

        # Level 1: tag + language + length
        mask_l1 = has_tag & (self.df["language"] == language) & (self.df["length"] == length)
        df_l1 = self.df[mask_l1]
        if not df_l1.empty:
            top_k = df_l1.sort_values(by="engagement", ascending=False).head(k)
            return top_k.to_dict(orient="records"), "tag + language + length"

        # Level 2: tag + language
        mask_l2 = has_tag & (self.df["language"] == language)
        df_l2 = self.df[mask_l2]
        if not df_l2.empty:
            top_k = df_l2.sort_values(by="engagement", ascending=False).head(k)
            return top_k.to_dict(orient="records"), "tag + language"

        # Level 3: tag only
        df_l3 = self.df[has_tag]
        if not df_l3.empty:
            top_k = df_l3.sort_values(by="engagement", ascending=False).head(k)
            return top_k.to_dict(orient="records"), "tag only"

        # Level 4: none
        return [], "none"
