from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


L1_META = ROOT_DIR / "ablation_study" / "A" / "datasets" / "L1_text_dataset_foreground_only" / "L1_200_metadata.json"
L2_META = ROOT_DIR / "ablation_study" / "A" / "datasets" / "L2_shapenet_foreground_only" / "L2_200_metadata.json"
L3_META = ROOT_DIR / "ablation_study" / "A" / "datasets" / "L3_selected_generated_foreground_only" / "L3_200_metadata.json"

SEED = 42
OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]


def attach_option_labels(record: dict) -> dict:
    labeled_options = []
    option_map = {}
    correct_option_label = None

    for label, option in zip(OPTION_LABELS, record["options"], strict=True):
        labeled_option = {"label": label, **option}
        labeled_options.append(labeled_option)
        option_map[label] = option["text"]
        if option["id"] == record["correct_option_id"]:
            correct_option_label = label

    record["options"] = labeled_options
    record["option_map"] = option_map
    record["correct_option_label"] = correct_option_label
    return record


def build_l1_vqa(samples: list[dict]) -> list[dict]:
    records = []
    for idx, sample in enumerate(samples):
        options = [sample["target_option"], *sample["distractors"]]
        random.Random(SEED + idx).shuffle(options)
        records.append(
            attach_option_labels(
                {
                "id": sample["id"],
                "question": "What text is revealed by the coherent motion in the challenge video?",
                "challenge_video": sample["video_path"],
                "options": [{"id": option["id"], "text": option["text"]} for option in options],
                "correct_option_id": sample["id"],
                "answer_text": sample["target_option"]["text"],
                "background_mode": sample["background_mode"],
                "source_text": sample["text"],
                }
            )
        )
    return records


def build_l2_vqa(samples: list[dict]) -> list[dict]:
    def option_id(option: dict) -> str:
        return f"{option['category']}::{option['model_id']}"

    records = []
    for idx, sample in enumerate(samples):
        options = [sample["target"], *sample["distractors"]]
        random.Random(SEED + idx).shuffle(options)
        records.append(
            attach_option_labels(
                {
                "id": sample["id"],
                "question": "Which object category is revealed by the structural motion in the challenge video?",
                "challenge_video": sample["video_path"],
                "options": [{"id": option_id(option), "text": option["category"]} for option in options],
                "correct_option_id": option_id(sample["target"]),
                "answer_text": sample["target"]["category"],
                "background_mode": sample["background_mode"],
                "target_category": sample["target"]["category"],
                }
            )
        )
    return records


def build_l3_vqa(samples: list[dict]) -> list[dict]:
    records = []
    for idx, sample in enumerate(samples):
        options = [sample["target_option"], *sample["distractors"]]
        random.Random(SEED + idx).shuffle(options)
        records.append(
            attach_option_labels(
                {
                "id": sample["id"],
                "question": "What action is the person performing in the challenge video?",
                "challenge_video": str(
                    (
                        Path("ablation_study")
                        / "A"
                        / "datasets"
                        / "L3_selected_generated_foreground_only"
                        / "library"
                        / sample["challenge_video"]
                    )
                ),
                "options": [{"id": option["id"], "text": option["category"]} for option in options],
                "correct_option_id": sample["id"],
                "answer_text": sample["target_option"]["category"],
                "background_mode": sample["background_mode"],
                "target_category": sample["category"],
                "local_relpath": sample["local_relpath"],
                "start_t": sample["start_t"],
                "end_t": sample["end_t"],
                }
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export six-way FG annotations with A-F labels.")
    parser.add_argument("--levels", nargs="+", choices=("L1", "L2", "L3"), default=["L1", "L2", "L3"])
    args = parser.parse_args()
    sources = {"L1": (L1_META, build_l1_vqa), "L2": (L2_META, build_l2_vqa), "L3": (L3_META, build_l3_vqa)}
    for level in dict.fromkeys(args.levels):
        metadata_path, builder = sources[level]
        samples = json.loads(metadata_path.read_text(encoding="utf-8"))
        output = metadata_path.with_name(f"{level}_200_vqa.json")
        output.write_text(json.dumps(builder(samples), indent=2), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
