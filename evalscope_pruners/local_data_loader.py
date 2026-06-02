"""local_data_loader: Helper to load LCB, AA-LCR and MMMU as local JSONL/JSON.

Keeps the pruners independent of evalscope server endpoints.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_local_dataset(
    path: str, format_hint: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load a dataset from a local file or directory.

    Supported formats: JSONL, JSON (list), JSON (dict with items_field).

    Args:
        path: Path to file or directory.
        format_hint: Optional hint ("jsonl" or "json").

    Returns:
        List of sample dictionaries.
    """
    p = Path(path)
    items: List[Dict[str, Any]] = []

    if p.is_file():
        return _load_file(p, format_hint)

    for f in sorted(p.glob("*")):
        if f.suffix.lower() in (".jsonl", ".json"):
            items.extend(_load_file(f, format_hint))
    return items


def _load_file(file_path: Path, format_hint: Optional[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    suffix = file_path.suffix.lower()
    is_jsonl = (format_hint == "jsonl") or (suffix == ".jsonl")

    if is_jsonl:
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        with open(file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, list):
            items = data[:]
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                items = data["data"][:]
            else:
                items.append(data)
    return items


def save_samples(samples: List[Dict[str, Any]], output_path: str, indent: int = 0) -> None:
    """Save samples to JSON (default) or JSONL."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        with open(p, "w", encoding="utf-8") as fp:
            for item in samples:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(p, "w", encoding="utf-8") as fp:
            json.dump(samples, fp, indent=indent, ensure_ascii=False)


def sample_counts(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count unique items by content."""
    seen: dict = {}
    for item in samples:
        key = str(item)[:80]
        seen[key] = seen.get(key, 0) + 1
    return {"total": len(samples), "unique": len(seen)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a local dataset.")
    parser.add_argument("path", help="Path to file or directory")
    parser.add_argument("--format", dest="fmt", choices=["json", "jsonl"], help="Force format")
    args = parser.parse_args()

    items = load_local_dataset(args.path, args.fmt)
    print(f"Loaded {len(items)} samples")
    if items:
        print("First sample keys:", list(items[0].keys()))
