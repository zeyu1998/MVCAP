import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT_DIR / "dataset" / "L3_selected_generated_rigid_drift"
DEFAULT_INPUT = DATASET_ROOT / "L3_rigid_drift_curated_metadata.json"
DEFAULT_OUTPUT = DATASET_ROOT / "L3_rigid_drift_curated_quiz_metadata.json"
CONFLICT_A = "kick"
CONFLICT_B = "exercise/training"


def load_curated_pool(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pool = []
    for item in data:
        pool.append(
            {
                "id": item["id"],
                "category": item["category"],
                "challenge_video": item["challenge_video"],
                "option_video": item["target_option"]["option_video"],
                "local_relpath": item.get("local_relpath"),
                "start_t": item.get("start_t"),
                "end_t": item.get("end_t"),
                "background_mode": item.get("background_mode", "rigid_drift"),
            }
        )
    return pool


def choose_distractor_categories(
    target_category: str,
    categories: list[str],
    rng: random.Random,
    num_distractors: int,
) -> list[str]:
    candidates = [cat for cat in categories if cat != target_category]

    if target_category == CONFLICT_A:
        candidates = [cat for cat in candidates if cat != CONFLICT_B]
        return rng.sample(candidates, num_distractors)

    if target_category == CONFLICT_B:
        candidates = [cat for cat in candidates if cat != CONFLICT_A]
        return rng.sample(candidates, num_distractors)

    base = [cat for cat in candidates if cat not in {CONFLICT_A, CONFLICT_B}]
    max_base = min(len(base), num_distractors)
    chosen = rng.sample(base, max_base)
    available_conflicts = [cat for cat in (CONFLICT_A, CONFLICT_B) if cat in candidates]
    if len(chosen) < num_distractors and available_conflicts:
        chosen.append(rng.choice(available_conflicts))
    while len(chosen) < num_distractors:
        fallback_pool = [cat for cat in candidates if cat not in chosen]
        if not fallback_pool:
            break
        chosen.append(rng.choice(fallback_pool))
    rng.shuffle(chosen)
    return chosen


def build_quiz_metadata(pool: list[dict], num_distractors: int) -> list[dict]:
    items_by_cat = defaultdict(list)
    for item in pool:
        items_by_cat[item["category"]].append(item)

    categories = sorted(items_by_cat.keys())
    if len(categories) < num_distractors + 1:
        raise ValueError(f"Need at least {num_distractors + 1} categories to build 1+{num_distractors} questions")

    final_metadata = []
    for item in pool:
        rng = random.Random(f"curated-quiz::{item['id']}")
        distractor_cats = choose_distractor_categories(item["category"], categories, rng, num_distractors)
        distractors = []
        for cat in distractor_cats:
            picked = rng.choice(items_by_cat[cat])
            distractors.append(
                {
                    "id": picked["id"],
                    "category": cat,
                    "option_video": picked["option_video"],
                }
            )

        final_metadata.append(
            {
                "id": item["id"],
                "category": item["category"],
                "challenge_video": item["challenge_video"],
                "target_option": {
                    "id": item["id"],
                    "category": item["category"],
                    "option_video": item["option_video"],
                },
                "distractors": distractors,
                "local_relpath": item["local_relpath"],
                "start_t": item["start_t"],
                "end_t": item["end_t"],
                "background_mode": item["background_mode"],
            }
        )

    return final_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble 1+5 L3 quiz metadata from curated rigid-drift samples.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-distractors", type=int, default=5)
    args = parser.parse_args()

    pool = load_curated_pool(args.input_path)
    quiz = build_quiz_metadata(pool, args.num_distractors)
    args.output_path.write_text(json.dumps(quiz, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output_path)
    print(f"items={len(quiz)}")


if __name__ == "__main__":
    main()
