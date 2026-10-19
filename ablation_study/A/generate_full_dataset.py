from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ablation_study.common import generate_foreground_only_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the full paired foreground-only ablation dataset from the official 200-sample metadata.")
    parser.add_argument("--limit", type=int, default=None, help="Optional small-limit smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if they already exist.")
    parser.add_argument("--levels", nargs="+", choices=("L1", "L2", "L3"), default=["L1", "L2", "L3"], help="Generate only the selected levels.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    outputs = generate_foreground_only_dataset(
        Path(__file__).resolve().parent,
        limit=args.limit,
        overwrite=args.overwrite,
        levels=tuple(dict.fromkeys(args.levels)),
    )
    print(f"Generated foreground-only dataset manifest: {outputs['manifest']}")
    for level in args.levels:
        print(f"{level} metadata: {outputs[level]}")


if __name__ == "__main__":
    main()
