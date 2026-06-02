"""run_pruners.py: CLI runner for DiscriminabilitySampler and ImageStressSampler.

Usage examples:
  python run_pruners.py part_a \
    --benchmarks data/lcb_v5.jsonl,data/aa_lcr.jsonl \
    --target 300 \
    --output results/part_a_subset.json

  python run_pruners.py part_b \
    --mmmu-dir data/mmmu \
    --target 1200 \
    --output results/part_b_subset.json
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark pruning samplers (Part A and Part B)."
    )
    sub = parser.add_subparsers(dest="task", required=True)

    p_a = sub.add_parser(
        "part_a",
        help="Prune LCB v5 / AA-LCR using DiscriminabilitySampler.",
    )
    p_a.add_argument(
        "--benchmarks",
        "-b",
        default="",
        help="Comma-separated paths to LCB/AA-LCR JSONL/JSON files or directories.",
    )
    p_a.add_argument(
        "--results-dir",
        "-r",
        default=None,
        help="Path to evalscope results directory (optional, loads scores).",
    )
    p_a.add_argument(
        "--target",
        "-t",
        type=int,
        default=300,
        help="Target size of pruned subset (default 300).",
    )
    p_a.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    p_b = sub.add_parser(
        "part_b",
        help="Prune MMMU 12K using ImageStressSampler.",
    )
    p_b.add_argument(
        "--mmmu-dir",
        "-m",
        default=None,
        help="Path to MMMU JSONL/JSON file or directory.",
    )
    p_b.add_argument(
        "--target",
        "-t",
        type=int,
        default=1200,
        help="Target size of pruned subset (default 1200).",
    )
    p_b.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )

    for p in (p_a, p_b):
        p.add_argument(
            "--output",
            "-o",
            required=False,
            help="Output path for pruned subset (JSON or JSONL).",
        )

    args = parser.parse_args()

    np = None
    try:
        import numpy as np  # type: ignore
    except ImportError:
        print("Error: numpy is required. Install: pip install numpy")
        return 1

    np.random.seed(args.seed)

    if args.task == "part_a":
        from evalscope_pruners import DiscriminabilitySampler

        sampler = DiscriminabilitySampler(
            data_path=args.benchmarks if args.benchmarks else None,
            results_dir=args.results_dir,
            target_size=args.target,
            seed=args.seed,
        )
        print(f"DiscriminabilitySampler initialized")
        print(f"  benchmarks: {args.benchmarks}")
        print(f"  results_dir: {args.results_dir}")
        print(f"  target_size: {args.target}")
        print(f"  seed: {args.seed}")
        if args.benchmarks:
            items = sampler()
        else:
            print("No --benchmarks path provided. Run with benchmark files to get a subset.")
            items = []
        if args.output and items:
            from evalscope_pruners.local_data_loader import save_samples
            save_samples(items, args.output)
            print(f"Saved {len(items)} samples to {args.output}")
        else:
            print(f"Selected {len(items)} samples")

    elif args.task == "part_b":
        from evalscope_pruners import ImageStressSampler

        sampler = ImageStressSampler(
            mmmu_dir=args.mmmu_dir,
            target_size=args.target,
            seed=args.seed,
        )
        print(f"ImageStressSampler initialized")
        print(f"  mmmu_dir: {args.mmmu_dir}")
        print(f"  target_size: {args.target}")
        print(f"  seed: {args.seed}")
        if args.mmmu_dir:
            items = sampler()
        else:
            print("No --mmmu-dir path provided. Run with MMMU data to get a subset.")
            items = []
        if args.output and items:
            from evalscope_pruners.local_data_loader import save_samples
            save_samples(items, args.output)
            print(f"Saved {len(items)} samples to {args.output}")
        else:
            print(f"Selected {len(items)} samples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
