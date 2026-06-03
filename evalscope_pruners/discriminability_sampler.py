"""DiscriminabilitySampler: Part A -- Prune LCB v5 / AA-LCR.

Fixes from code review:
  1. Match-count stratification: group by exact # models passing (0/1/2/3),
     not continuous ranges that collapse to empty buckets with 3 binary models.
  2. Metadata tie-breaking: use judge reasoning length, generation length,
     and failure-mode diversity instead of variance (identical for all samples
     in 1-pass and 2-pass strata with binary scores).
  3. No random fallback: all selection is deterministic from embedded metadata.
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from evalscope_pruners.base import Sampler


class DiscriminabilitySampler(Sampler):
    """
    Prune LCB v5 and/or AA-LCR into a minimal, informative probe set.

    Stratification: With 3 models producing binary {0,1} scores, each sample
    has a match count in {0,1,2,3}. We allocate probe slots across these
    strata proportionally, guaranteeing coverage of the full difficulty range.

    Tie-breaking within strata uses embedded metadata:
      - LCB: generation length + failure-mode categorization (timeout/syntax/runtime)
      - AA-LCR: judge reasoning length + judge confidence (LLM judge non-determinism)
    This is orthogonal to model scores and works for any model ensemble.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        target_size: int = 300,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.results_dir = results_dir
        self.target_size = max(50, int(target_size))
        self.seed = seed
        np.random.seed(seed)
        self._model_scores: Dict[str, Dict[str, float]] = {}
        self._model_names: List[str] = []

    # === Data Loading ===

    def _load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        items = []
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            return items
        with open(filepath, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _load_predictions(self) -> List[Dict[str, Any]]:
        if not self.data_path:
            return []
        paths = [p.strip() for p in self.data_path.split(",") if p.strip()]
        items: List[Dict[str, Any]] = []
        for p in paths:
            path = Path(p)
            if path.is_file():
                items.extend(self._load_jsonl(str(path)))
            elif path.is_dir():
                for jsonl in sorted(path.glob("*.jsonl")):
                    items.extend(self._load_jsonl(str(jsonl)))
        return items

    def _load_model_scores(self) -> None:
        if not self.results_dir:
            return
        for _rdir, model_dir in [(r, m) for r in [d.strip() for d in self.results_dir.split(",") if d.strip() and os.path.isdir(d.strip())] for m in os.listdir(r)]:
            model_path = os.path.join(_rdir, model_dir)
            if not os.path.isdir(model_path):
                continue
            for subdir in os.listdir(model_path):
                sub_path = os.path.join(model_path, subdir)
                if not os.path.isdir(sub_path):
                    continue
                eval_path = os.path.join(sub_path, "eval")
                if not os.path.isdir(eval_path):
                    continue
                answers_file = os.path.join(eval_path, "answers.json")
                if os.path.exists(answers_file):
                    with open(answers_file, "r", encoding="utf-8") as fp:
                        answers = json.load(fp)
                    for ans in answers:
                        model_name = str(ans.get("model_name", ans.get("model", "")))
                        idx_key = str(ans.get("index", ans.get("choice", "")))
                        pred = ans.get("prediction", "")
                        gold = ans.get("answer", ans.get("gold", ""))
                        score = 1.0 if str(pred) == str(gold) else 0.0
                        if model_name not in self._model_scores:
                            self._model_scores[model_name] = {}
                        self._model_scores[model_name][idx_key] = score
        self._model_names = sorted(self._model_scores.keys())

    # === Metadata Extraction ===

    def _lcb_meta(self, item: Dict[str, Any]) -> Dict[str, float]:
        pred = item.get("prediction", "") or item.get("response", "")
        meta = {"gen_length": float(len(str(pred)))}
        exec_info = item.get("execution", item.get("exec_result", item.get("stdout", {})))
        if isinstance(exec_info, dict):
            err = exec_info.get("error_type", exec_info.get("error", exec_info.get("status", "none")))
        else:
            err = str(exec_info)
        if "timeout" in err.lower() or err in ("TLE", "time_limit_exceeded"):
            meta["failure_timeout"] = 1.0
            meta["failure_syntax"] = 0.0
        elif "syntax" in err.lower() or "parse" in err.lower():
            meta["failure_timeout"] = 0.0
            meta["failure_syntax"] = 1.0
        else:
            meta["failure_timeout"] = 0.0
            meta["failure_syntax"] = 0.0
        return meta

    def _aalcr_meta(self, item: Dict[str, Any]) -> Dict[str, float]:
        review = item.get("review", {})
        if isinstance(review, dict):
            reasoning = review.get("reasoning", review.get("explanation", ""))
            conf = review.get("confidence", 0.5)
        elif isinstance(review, str):
            reasoning = review
            conf = 0.5
        else:
            reasoning = ""
            conf = 0.5
        score_dict = item.get("sample_score", item.get("score", {}))
        if isinstance(score_dict, dict):
            conf = score_dict.get("confidence", conf)
        return {
            "judge_reasoning_len": float(len(str(reasoning))),
            "judge_confidence": float(conf),
            "judge_logprob": float(score_dict.get("logprob", -1.0) if isinstance(score_dict, dict) else -1.0),
        }

    def _generic_meta(self, item: Dict[str, Any]) -> Dict[str, float]:
        pred = item.get("prediction", item.get("response", ""))
        return {"gen_length": float(len(str(pred)))}

    # === Match-Count Stratification ===

    def _match_counts(
        self, items: List[Dict[str, Any]]
    ) -> List[Tuple[int, List[float], int]]:
        """
        Compute (match_count, per_model_scores, idx) for each sample.
        match_count is in {0,1,2,3} with 3 binary models.
        """
        results = []
        for idx, item in enumerate(items):
            idx_key = str(item.get("index", item.get("choice", item.get("id", idx))))
            scores = [self._model_scores.get(m, {}).get(idx_key, 0.0) for m in self._model_names]
            mc = sum(1 for s in scores if s > 0.5)
            results.append((mc, scores, idx))
        return results
        
    def fit(self, items: List[Dict[str, Any]], **kwargs) -> "DiscriminabilitySampler":
        self._load_model_scores()
        self._items = list(items)
        self._match_data = self._match_counts(self._items)
        self._n_models = len(self._model_names) if self._model_names else 1
        n_items = len(items)
        if n_items > 0:
            self._n_per_stratum = max(1, self.target_size // 4)
            leftover = self.target_size - (self._n_per_stratum * 4)
            if leftover > 0:
                self._n_per_stratum = self._n_per_stratum + leftover // 4
        else:
            self._n_per_stratum = 0
        return self

    def __call__(
        self,
        items: Optional[Iterable[Dict[str, Any]]] = None,
        target_size: Optional[int] = None,
        dataset: str = "generic",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if items is None and not hasattr(self, "_items") and self.data_path:
            items = self._load_predictions()
        items_list = list(items) if items is not None else list(self._items)
        if not items_list:
            return []
        if not hasattr(self, "_match_data"):
            self.fit(items_list)
        tgt = target_size if target_size is not None else self.target_size
        n_per = max(1, tgt // 4)
        leftover = tgt - (n_per * 4)
        extra = leftover
        # Route meta extractor by dataset tag
        if "lcb" in dataset.lower() or "livecode" in dataset.lower():
            meta_fn = self._lcb_meta
        elif "aalcr" in dataset.lower() or "aa" in dataset.lower():
            meta_fn = self._aalcr_meta
        else:
            meta_fn = self._generic_meta
        # Build strata: stratum_id -> [(item_idx, meta_dict)]
        strata: Dict[int, List[Tuple[int, Dict[str, float]]]] = defaultdict(list)
        for mc, scores, idx in self._match_data:
            strata[mc].append((idx, meta_fn(items_list[idx])))
        # Select from each stratum by metadata
        selected: List[int] = []
        for mc in [0, 1, 2, 3]:
            bucket = strata.get(mc, [])
            if not bucket:
                continue
            n_take = n_per + (1 if mc < extra else 0)
            n_take = min(n_take, len(bucket))
            if n_take == len(bucket):
                selected.extend([i for i, _ in bucket])
                continue
            # LCB: sort by gen_length desc, then by failure-mode diversity
            # AA-LCR: sort by judge_reasoning_len desc, then by confidence
            # Both are deterministic and metadata-driven
            relates_to_lcb = ("lcb" in dataset.lower())
            if relates_to_lcb:
                bucket_sorted = sorted(bucket, key=lambda x: (
                    -x[1].get("gen_length", 0),
                    -x[1].get("failure_timeout", 0),
                    -x[1].get("failure_syntax", 0),
                ))
            else:
                bucket_sorted = sorted(bucket, key=lambda x: (
                    -x[1].get("judge_reasoning_len", 0),
                    -x[1].get("judge_confidence", 0),
                    -x[1].get("judge_logprob", 0),
                ))
            selected.extend([i for i, _ in bucket_sorted[:n_take]])
        selected = sorted(set(selected))
        return [items_list[i] for i in selected if i < len(items_list)]
