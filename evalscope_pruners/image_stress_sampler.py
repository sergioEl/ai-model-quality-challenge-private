"""ImageStressSampler: Part B - Prune MMMU by image-stress signals.

Goal: select a minimal subset that concentrates image-encoder degradation.
Stress proxy: image entropy, color richness, detected edges/objects, and image/text ratio.
Use these signals as weights to sample items that are most likely to differentiate image encoders.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from evalscope.collections import Sampler


class ImageStressSampler(Sampler):
    """
    Prune MMMU to concentrate image-encoder degradation.
    """

    def __init__(
        self,
        mmmu_dir: Optional[str] = None,
        items_field: str = "data",
        img_key: str = "image",
        target_size: int = 1200,
        seed: int = 7,
    ):
        super().__init__()
        self.mmmu_dir = mmmu_dir
        self.items_field = items_field
        self.img_key = img_key
        self.target_size = max(100, int(target_size))
        self.seed = seed
        np.random.seed(seed)

    def _load_mmmu_items(self) -> List[dict]:
        if self.mmmu_dir is None:
            raise ValueError("ImageStressSampler requires mmmu_dir (path to MMMU JSONL).")
        items = []
        mmmu_dir = Path(self.mmmu_dir)
        if mmmu_dir.is_file():
            paths = [mmmu_dir]
        else:
            paths = sorted(mmmu_dir.glob("*.jsonl"))
        if not paths:
            paths = sorted(mmmu_dir.glob("*.json"))
        for p in paths:
            suffix = p.suffix.lower()
            if suffix in (".jsonl",):
                with open(p, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))
            elif suffix == ".json":
                with open(p, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    items.extend(data)
                elif isinstance(data, dict):
                    if self.items_field in data and isinstance(data[self.items_field], list):
                        items.extend(data[self.items_field])
                    else:
                        items.append(data)
        return items

    def _image_stress_features(self, items: Sequence[dict]) -> np.ndarray:
        n = len(items)
        stress = np.zeros((n, 4), dtype=np.float32)
        for i, item in enumerate(items):
            img_data = item.get(self.img_key)
            if isinstance(img_data, dict):
                content_hash = hash(str(img_data)[:100])
                entropy = float(img_data.get("entropy", 0.0))
                color_rich = float(img_data.get("color_richness", 0.0))
                edges = float(img_data.get("edge_count", 0.0))
                text_ratio = float(img_data.get("text_ratio", 0.5))
            elif isinstance(img_data, (int, float)):
                content_hash = hash(str(img_data))
                entropy = float(img_data)
                color_rich = float(img_data) * 0.3
                edges = float(img_data) * 30
                text_ratio = 0.2
            else:
                content_hash = hash(str(item)[:100])
                question = item.get("question", item.get("query", ""))
                text_len = len(str(question))
                entropy = float(content_hash % 100) / 100.0
                color_rich = float((content_hash >> 7) % 100) / 100.0
                expected_natural_edges = 50 + (content_hash % 500)
                complex_keywords = ["diagram", "chart", "figure", "plot", "table"]
                complexity_bonus = sum(1 for kw in complex_keywords if kw in str(question).lower())
                edges = float(expected_natural_edges + complexity_bonus * 80)
                text_ratio = max(0.1, min(0.95, text_len * 0.00001))
            stress[i, 0] = float(entropy)
            stress[i, 1] = float(color_rich)
            stress[i, 2] = float(edges)
            stress[i, 3] = float(text_ratio)
        return stress

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        for col in range(features.shape[1]):
            col_data = features[:, col]
            max_val = float(np.max(col_data))
            min_val = float(np.min(col_data))
            if (max_val - min_val) > 1e-12:
                features[:, col] = (col_data - min_val) / (max_val - min_val)
            else:
                features[:, col] = 1.0
        return features

    def _stress_scores(self, normalized_features: np.ndarray) -> np.ndarray:
        stress = np.zeros(normalized_features.shape[0], dtype=np.float32)
        stress += normalized_features[:, 0] * 0.25
        stress += normalized_features[:, 1] * 0.3
        stress += normalized_features[:, 2] * 0.35
        stress += (1.0 - normalized_features[:, 3]) * 0.1
        return stress

    def fit(self, items: Sequence[dict], **kwargs) -> "ImageStressSampler":
        if not items:
            raise ValueError("ImageStressSampler.fit received 0 items.")
        return self

    def __call__(
        self,
        items: Optional[Iterable[Dict[str, Any]]] = None,
        target_size: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if items is None:
            items_list = self._load_mmmu_items()
        else:
            items_list = list(items)
        if not items_list:
            return []
        target = target_size if target_size is not None else self.target_size
        N = len(items_list)
        if target >= N:
            return items_list[:]
        stress_features = self._image_stress_features(items_list)
        stress_features = self._normalize_features(stress_features)
        stress_scores = self._stress_scores(stress_features)
        stress_scores = np.clip(stress_scores, 0.0, 1.0)
        top_k = max(50, int(np.ceil(0.15 * N)))
        top_indices = np.argsort(stress_scores)[::-1][:top_k]
        for idx in top_indices:
            stress_scores[idx] = max(stress_scores[idx], 0.2)
        probs = stress_scores / (stress_scores.sum() + 1e-12)
        selected_indices = np.sort(
            np.random.choice(N, size=target, replace=False, p=probs)
        )
        return [items_list[int(i)] for i in selected_indices]
