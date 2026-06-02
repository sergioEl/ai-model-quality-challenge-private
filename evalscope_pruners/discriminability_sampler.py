"""DiscriminabilitySampler: Part A - Prune LCB v5 / AA-LCR.

Selection criterion: approximate Leave-One-Out discriminative value.
A sample is useful if it has one model clearly correct while another is clearly wrong.
Keeps samples where model scores are polarized and one score is high near 1.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from evalscope.collections import Sampler


class DiscriminabilitySampler(Sampler):
    """
    Prune LCB v5 / AA-LCR to a small, informative subset.

    Uses an approximate discriminative-value objective:
      score_i = max(|s_a - s_b|, |s_b - s_c|, |s_a - s_c|)
      reward_i = score_i * max(s_a, s_b, s_c)
    This favors samples where models disagree AND at least one performs well.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        target_size: int = 300,
        results_prefix: str = "results/evalscope_run",
        benchmark_builds: str = "jsonl",
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.results_dir = results_dir
        self.target_size = max(50, int(target_size))
        self.results_prefix = results_prefix
        self.benchmark_builds = benchmark_builds
        self.seed = seed

    def _load_jsonl_items(self, filepath: str) -> List[dict]:
        filepath = os.path.abspath(filepath)
        with open(filepath, "r", encoding="utf-8") as fp:
            lines = [line.strip() for line in fp]
        items = []
        for line in lines:
            if not line:
                continue
            items.append(json.loads(line))
        return items

    def _load_sample_rows(self, build_tokens: str) -> List[dict]:
        rows: List[dict] = []
        tokens = [tok.strip() for tok in build_tokens.split(",") if tok.strip()]
        for tok in tokens:
            base_path = Path(tok)
            if base_path.is_dir():
                file_dir = base_path
            else:
                file_dir = base_path.parent
                if not file_dir.exists():
                    file_dir = Path(".")
            for jsonl_path in file_dir.glob("*.jsonl"):
                items = self._load_jsonl_items(str(jsonl_path))
                for item in items:
                    sample = item.get("sample") or item
                    sample["_data_parent"] = tok
                    rows.append(sample)
        return rows

    def _load_model_scores(
        self, items: Sequence[dict]
    ) -> Tuple[List[Dict[str, float]], Dict[str, str]]:
        if self.data_path and os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            model_scores: List[Dict[str, float]] = data.get("model_scores", [])
            model_model: Dict[str, str] = data.get("model_model", {})
        else:
            model_scores = []
            model_model = {}
        if self.results_dir and os.path.exists(self.results_dir):
            results: Dict[str, Dict[str, float]] = {}
            for entry in os.listdir(self.results_dir):
                full_entry = os.path.join(self.results_dir, entry)
                if not os.path.isdir(full_entry):
                    continue
                for tag in os.listdir(full_entry):
                    full_tag = os.path.join(full_entry, tag)
                    if not os.path.isdir(full_tag):
                        continue
                    eval_dir = os.path.join(full_tag, "eval")
                    if not os.path.isdir(eval_dir):
                        continue
                    answers_json = os.path.join(eval_dir, "answers.json")
                    if not os.path.exists(answers_json):
                        continue
                    with open(answers_json, "r", encoding="utf-8") as fp:
                        answers = json.load(fp)
                    per_rev: Dict[str, Dict[str, float]] = {}
                    for ans in answers:
                        model_name = str(ans.get("model_name", ans.get("model", "")))
                        prediction = ans.get("prediction", "")
                        answer_str = ans.get("answer", "")
                        if prediction is None:
                            prediction = ""
                        if answer_str is None:
                            answer_str = ""
                        rev_id = str(ans.get("choice", ans.get("review_id", "")))
                        pred_score = 1.0 if str(prediction) == str(answer_str) else 0.0
                        if model_name not in per_rev:
                            per_rev[model_name] = {}
                        per_rev[model_name][rev_id] = float(pred_score)
                    results[tag] = per_rev
            if results:
                all_models = set()
                for per_rev in results.values():
                    all_models.update(per_rev.keys())
                merged: Dict[str, Dict[str, float]] = {}
                for model_name in all_models:
                    merged[model_name] = {}
                for per_rev in results.values():
                    for model_name, rev_dict in per_rev.items():
                        for rev_id, score in rev_dict.items():
                            existing = merged.get(model_name, {}).get(rev_id)
                            if existing is not None:
                                merged[model_name][rev_id] = existing + score
                            else:
                                merged[model_name][rev_id] = score
                for model_name in all_models:
                    rev_dict = merged.get(model_name, {})
                    for k, v in rev_dict.items():
                        rev_dict[k] = max(0.0, min(1.0, v))
                json_obj: Dict[str, Any] = {}
                json_obj["model_model"] = model_model
                json_obj["model_scores"] = []
                for model_name, rev_dict in merged.items():
                    ScoreEntry = {"model": model_name, "scores": rev_dict}
                    json_obj["model_scores"].append(ScoreEntry)
                self.data_path = None
                return self._load_model_scores([])
        if model_scores:
            merged = {}
            for entry in model_scores:
                model_name = entry.get("model", entry.get("model_name", ""))
                scores = entry.get("scores", {})
                if model_name not in merged:
                    merged[model_name] = {}
                for k, v in scores.items():
                    merged[model_name][k] = float(v)
            return list(merged.values()), model_model
        return [], {}

    def _pairwise_score_matrix(self, items: Sequence[dict]) -> np.ndarray:
        n = len(items)
        S = np.zeros((n, n), dtype=np.float32)
        for i, item in enumerate(items):
            model_answer = item.get("model_answer") or item.get("answer", "")
            for j, other in enumerate(items):
                if i == j:
                    continue
                other_answer = other.get("model_answer") or other.get("answer", "")
                if str(model_answer) == str(other_answer):
                    S[i, j] = 1.0
                else:
                    S[i, j] = 0.0
        return S

    def _build_data_table(
        self, items: Sequence[dict]
    ) -> Tuple[List[dict], List[int], Dict[str, int]]:
        index_map: Dict[int, int] = {}
        data_rows: List[dict] = []
        used: Dict[str, int] = {}
        for orig_idx, item in enumerate(items):
            revision_id = str(item.get("choice", item.get("review_id", "")))
            content = item.get("query", item.get("input", item.get("text", "")))
            model_answer = item.get("model_answer") or item.get("answer", "")
            data_rows.append({"content": content, "model_answer": model_answer})
            used[revision_id] = orig_idx
            index_map[orig_idx] = len(data_rows) - 1
        return data_rows, [index_map[i] for i in range(len(items))], used

    def _discriminative_values(
        self, items: Sequence[dict], score_matrix: np.ndarray
    ) -> np.ndarray:
        n = len(items)
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            max_gap = 0.0
            max_acc = 0.0
            for j in range(i + 1, n):
                gap = abs(score_matrix[i, j] - 0.5)
                if score_matrix[i, j] > 0.5:
                    candidate = min(1.0, score_matrix[i, j])
                else:
                    candidate = 0.0
                acc = max(score_matrix[i, j], candidate)
                if gap > max_gap or (gap == max_gap and acc > max_acc):
                    max_gap = gap
                    max_acc = acc
            if max_gap > 0:
                scores[i] = max_gap * max_acc
            else:
                scores[i] = 0.0
        return scores

    def fit(self, items: Sequence[dict], **kwargs) -> "DiscriminabilitySampler":
        if not items:
            raise ValueError("DiscriminabilitySampler.fit received 0 items.")
        score_rows, model_model = self._load_model_scores(items)
        self._score_rows = score_rows
        self._model_model = model_model
        return self

    def _score_components(self, items: Sequence[dict]) -> np.ndarray:
        rows_per_item = [self._score_rows[i] if i < len(self._score_rows) else {} for i in range(len(items))]
        target_size = self.target_size
        n = len(items)
        if n == 0:
            return np.array([], dtype=np.float32)
        if target_size >= n:
            return np.ones(n, dtype=np.float32)
        keep_per = max(1, int(np.ceil(0.1 * n)))
        components = np.zeros(n, dtype=np.float32)
        scores_matrix = self._pairwise_score_matrix(items)
        disc_values = self._discriminative_values(items, scores_matrix)
        top_k_by_disc = max(keep_per, int(np.ceil(0.05 * n)))
        top_indices = np.argsort(disc_values)[::-1][:top_k_by_disc]
        components[top_indices] = np.maximum(components[top_indices], disc_values[top_indices])
        model_rows = self._score_rows
        if model_rows:
            model_scores = {}
            for entry in model_rows:
                model_name = entry.get("model", entry.get("model_name", ""))
                scores_dict = entry.get("scores", {})
                model_scores[model_name] = scores_dict
            if model_scores:
                model_names = sorted(model_scores.keys())
                num_models = len(model_names)
                for i, item in enumerate(items):
                    content = item.get("query", item.get("input", item.get("text", "")))
                    content_hash = hash(str(content)[:100])
                    model_vals_i = []
                    for model_name in model_names:
                        all_scores = model_scores[model_name]
                        if content_hash in all_scores:
                            model_vals_i.append(all_scores[content_hash])
                        elif i < len(all_scores):
                            model_vals_i.append(all_scores[i])
                        else:
                            model_vals_i.append(0.5)
                    max_model = max(model_vals_i)
                    min_model = min(model_vals_i)
                    spread = max_model - min_model
                    if spread > 0 and max_model > 0:
                        components[i] += spread * max_model
        if np.any(components):
            components = components / (np.max(components) + 1e-9)
        components = np.clip(components, 0.0, 1.0)
        top_indices = np.argsort(components)[::-1]
        for idx in top_indices[:keep_per]:
            components[idx] = max(components[idx], 0.2)
        return components

    def __call__(
        self,
        items: Optional[Iterable[Dict[str, Any]]] = None,
        target_size: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if items is None:
            if self.data_path is None or not os.path.exists(str(self.data_path)):
                raise ValueError("DiscriminabilitySampler requires data_path if items is not provided.")
            build_tokens = str(self.data_path)
            items_list = self._load_sample_rows(build_tokens)
        else:
            items_list = list(items)
        if not items_list:
            return []
        target = target_size if target_size is not None else self.target_size
        N = len(items_list)
        if target >= N:
            return items_list[:]
        if not hasattr(self, "_score_rows"):
            self.fit(items_list)
        components = self._score_components(items_list)
        if isinstance(components, (list, tuple)):
            components = np.array(components, dtype=np.float32)
        probs = np.asarray(components, dtype=np.float32)
        probs = probs / (probs.sum() + 1e-9)
        selected_indices = np.sort(
            np.random.choice(N, size=target, replace=False, p=probs)
        )
        return [items_list[int(i)] for i in selected_indices]
