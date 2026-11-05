"""Passage-aware, deterministic WikiText token windows."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Window:
    input_ids: list[int]
    window_id: str
    passage_id: str
    split: str


def windows_from_rows(rows, tokenizer, split, count, window_size=128):
    """Tokenize each article separately and retain complete, nonoverlapping windows.

    WikiText article headings are level-one headings; rows before the first
    heading form a separate passage. Incomplete article tails are discarded.
    """
    if count < 0 or window_size < 2:
        raise ValueError("count must be nonnegative and window_size at least 2")
    articles = []
    current = []
    for row in rows:
        text = row["text"]
        if re.match(r"^\s*=\s*[^=\n]+?\s*=\s*$", text):
            if current:
                articles.append("\n".join(current))
            current = [text]
        elif text.strip():
            current.append(text)
    if current:
        articles.append("\n".join(current))
    windows = []
    for article_index, article in enumerate(articles):
        ids = tokenizer(article, add_special_tokens=False)["input_ids"]
        passage_id = f"{split}-article-{article_index:05d}"
        for offset in range(0, len(ids) - window_size + 1, window_size):
            if len(windows) == count:
                return windows
            windows.append(Window(list(ids[offset:offset + window_size]),
                                  f"{passage_id}-token-{offset:07d}", passage_id, split))
    if len(windows) < count:
        raise ValueError(f"{split} contains only {len(windows)} complete windows; requested {count}")
    return windows


def load_windows(tokenizer, split, count, window_size=128, revision=None):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid dataset split: {split}")
    from datasets import load_dataset
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, revision=revision)
    return windows_from_rows(rows, tokenizer, split, count, window_size)
