import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
BABEL_DIR = ROOT_DIR / "dataset" / "babel_v1.0_release"
CMU_ROOT = ROOT_DIR / "dataset" / "L3_raw" / "CMU"
OUTPUT_ROOT = ROOT_DIR / "dataset" / "L3_babel"
SELECTION_PATH = OUTPUT_ROOT / "l3_14class_selection.json"
SUMMARY_PATH = OUTPUT_ROOT / "l3_14class_selection_summary.json"
TARGET_FPS = 30

BABEL_FILES = ["train.json", "val.json", "test.json"]
TARGET_ACTIONS = [
    "walk",
    "run",
    "jump",
    "kick",
    "punch",
    "dance",
    "bend",
    "crouch",
    "stand up",
    "wave",
    "turn",
    "golf",
    "exercise/training",
]


def extract_labels(item: dict) -> list[dict]:
    labels = []
    if item.get("frame_ann") and item["frame_ann"].get("labels"):
        labels.extend(item["frame_ann"]["labels"])
    if item.get("seq_ann") and item["seq_ann"].get("labels"):
        labels.extend(item["seq_ann"]["labels"])
    return labels


def normalize_local_relpath(feat_p: str) -> str | None:
    if feat_p.startswith("CMU/CMU/"):
        return feat_p[len("CMU/") :]
    return None


def score_candidate(candidate: dict, target_duration: float) -> tuple:
    # Prefer high post-downsample frame counts first, then prefer durations close to target,
    # then prefer exact proc/raw label matches.
    exact_bonus = 0 if candidate["action"] in candidate["act_cat"] else 1
    proc_exact = 0 if candidate["proc_label"].strip().lower() == candidate["action"] else 1
    return (
        -candidate["frames_after_stride"],
        abs(candidate["duration"] - target_duration),
        exact_bonus,
        proc_exact,
        candidate["local_relpath"],
        candidate["start_t"],
    )


def collect_candidates(min_duration: float, max_duration: float, min_frames_after_stride: int) -> dict[str, list[dict]]:
    by_action = defaultdict(list)
    seen = set()
    clip_meta_cache: dict[str, tuple[float, int]] = {}

    for filename in BABEL_FILES:
        babel_path = BABEL_DIR / filename
        if not babel_path.exists():
            continue

        with babel_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for babel_id, item in data.items():
            feat_p = item.get("feat_p", "")
            local_rel = normalize_local_relpath(feat_p)
            if not local_rel:
                continue

            local_path = ROOT_DIR / "dataset" / "L3_raw" / local_rel
            if not local_path.exists():
                continue

            local_key = str(local_path)
            if local_key not in clip_meta_cache:
                data = np.load(local_path)
                fps = float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 120.0
                num_frames = int(data["poses"].shape[0])
                clip_meta_cache[local_key] = (fps, num_frames)
            fps, num_frames = clip_meta_cache[local_key]
            clip_duration = num_frames / fps if fps > 0 else 0.0

            for label in extract_labels(item):
                act_cats = [x.lower() for x in (label.get("act_cat") or []) if isinstance(x, str)]
                for action in TARGET_ACTIONS:
                    if action not in act_cats:
                        continue

                    start_t = float(label.get("start_t", 0.0))
                    end_t = float(label.get("end_t", item.get("dur", 0.0)))
                    start_t = max(0.0, start_t)
                    end_t = min(end_t, clip_duration)
                    duration = max(0.0, end_t - start_t)
                    if duration < min_duration or duration > max_duration:
                        continue
                    start_idx = max(0, int(np.floor(start_t * fps)))
                    end_idx = max(start_idx + 1, int(np.ceil(end_t * fps)))
                    end_idx = min(end_idx, num_frames)
                    stride = max(1, int(fps // TARGET_FPS))
                    frames_after_stride = len(np.arange(start_idx, end_idx, stride))
                    if frames_after_stride < min_frames_after_stride:
                        continue

                    key = (
                        action,
                        local_rel,
                        round(start_t, 3),
                        round(end_t, 3),
                        str(label.get("proc_label", "")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    by_action[action].append(
                        {
                            "action": action,
                            "babel_split": filename,
                            "babel_id": babel_id,
                            "babel_sid": item.get("babel_sid"),
                            "feat_p": feat_p,
                            "local_relpath": local_rel,
                            "local_path": str(local_path),
                            "subject": Path(local_rel).parent.name,
                            "clip_id": Path(local_rel).stem.replace("_poses", "").replace("_stageii", ""),
                            "duration": duration,
                            "src_fps": fps,
                            "frames_after_stride": frames_after_stride,
                            "start_t": start_t,
                            "end_t": end_t,
                            "proc_label": str(label.get("proc_label", "")),
                            "raw_label": str(label.get("raw_label", "")),
                            "act_cat": act_cats,
                        }
                    )
                    break

    return by_action


def select_balanced_set(
    candidates: dict[str, list[dict]],
    per_class: int,
    target_duration: float,
    allow_shortfall: bool,
) -> dict[str, list[dict]]:
    selection = {}
    globally_reserved_files = set()
    action_order = sorted(
        TARGET_ACTIONS,
        key=lambda action: (
            len({item["local_relpath"] for item in candidates[action]}),
            len(candidates[action]),
            action,
        ),
    )

    for action in action_order:
        items = sorted(candidates[action], key=lambda x: score_candidate(x, target_duration))
        requested = min(per_class, len(items)) if allow_shortfall else per_class

        if len(items) < per_class and not allow_shortfall:
            raise ValueError(f"Action '{action}' only has {len(items)} candidates, fewer than requested {per_class}.")

        chosen = []
        action_reserved_files = set()

        for item in items:
            local_relpath = item["local_relpath"]
            if local_relpath in globally_reserved_files:
                continue
            if local_relpath in action_reserved_files:
                continue
            chosen.append(item)
            action_reserved_files.add(local_relpath)
            if len(chosen) == requested:
                break

        if len(chosen) < requested:
            chosen_keys = {
                (
                    c["local_relpath"],
                    round(c["start_t"], 3),
                    round(c["end_t"], 3),
                    c["proc_label"],
                )
                for c in chosen
            }
            for item in items:
                local_relpath = item["local_relpath"]
                key = (
                    local_relpath,
                    round(item["start_t"], 3),
                    round(item["end_t"], 3),
                    item["proc_label"],
                )
                if key in chosen_keys:
                    continue
                if local_relpath in globally_reserved_files and local_relpath not in action_reserved_files:
                    continue
                chosen.append(item)
                action_reserved_files.add(local_relpath)
                chosen_keys.add(key)
                if len(chosen) == requested:
                    break

        if len(chosen) < requested and not allow_shortfall:
            raise ValueError(
                f"Action '{action}' could only reserve {len(chosen)} segments "
                f"without sharing npz files across categories."
            )

        globally_reserved_files.update(action_reserved_files)
        selection[action] = chosen

    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced high-frame-priority L3 selection from CMU+BABEL.")
    parser.add_argument("--per-class", type=int, default=20, help="Number of segments per action class.")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Minimum segment duration in seconds. Default keeps only segments with duration >= 3s.")
    parser.add_argument("--max-duration", type=float, default=8.0, help="Maximum segment duration in seconds.")
    parser.add_argument("--target-duration", type=float, default=3.0, help="Preferred segment duration for ranking.")
    parser.add_argument(
        "--min-frames-after-stride",
        type=int,
        default=1,
        help="Hard lower bound on the number of frames remaining after downsampling to 30 FPS.",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Allow classes with fewer than --per-class high-frame candidates to keep a smaller count instead of failing.",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates(args.min_duration, args.max_duration, args.min_frames_after_stride)
    selection = select_balanced_set(candidates, args.per_class, args.target_duration, args.allow_shortfall)

    flat = []
    summary = {}
    for action, items in selection.items():
        flat.extend(items)
        summary[action] = {
            "selected_segments": len(items),
            "unique_files": len({item["local_relpath"] for item in items}),
            "min_duration": round(min(item["duration"] for item in items), 3),
            "max_duration": round(max(item["duration"] for item in items), 3),
            "mean_duration": round(sum(item["duration"] for item in items) / len(items), 3),
            "min_frames_after_stride": min(item["frames_after_stride"] for item in items),
            "max_frames_after_stride": max(item["frames_after_stride"] for item in items),
            "mean_frames_after_stride": round(sum(item["frames_after_stride"] for item in items) / len(items), 2),
        }

    SELECTION_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved balanced selection to {SELECTION_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    for action in TARGET_ACTIONS:
        info = summary[action]
        print(
            f"  {action}: {info['selected_segments']} segments, "
            f"{info['unique_files']} unique files, "
            f"mean_dur={info['mean_duration']}, "
            f"mean_frames={info['mean_frames_after_stride']}"
        )
    print(f"Total selected segments: {len(flat)}")


if __name__ == "__main__":
    main()
