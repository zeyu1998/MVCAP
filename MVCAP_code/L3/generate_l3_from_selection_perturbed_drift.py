import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch

from .generate_l3_from_selection_rigid_drift import (
    CHALLENGE_DRAW_SCALE,
    CHALLENGE_HEIGHT,
    CHALLENGE_OFFSET_Y,
    BACKGROUND_POINT_COUNT,
    CHALLENGE_WIDTH,
    JOINT_INDICES_17,
    MODEL_ROOT,
    OPTION_HEIGHT,
    OPTION_WIDTH,
    ROOT_DIR,
    SELECTION_PATH,
    convert_to_h264,
    load_segment,
    render_option,
    sanitize_name,
)


OUTPUT_DIR = ROOT_DIR / "dataset" / "L3_selected_generated_perturbed_drift"
LIB_DIR = OUTPUT_DIR / "library"
METADATA_PATH = OUTPUT_DIR / "L3_metadata.json"
FPS = 30


def init_random_noise(width: int, height: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.stack(
        [
            rng.uniform(0, width, size=count),
            rng.uniform(0, height, size=count),
        ],
        axis=1,
    ).astype(np.float32)
    vel = np.stack(
        [
            rng.normal(-0.35, 0.28, size=count),
            rng.normal(0.0, 0.14, size=count),
        ],
        axis=1,
    ).astype(np.float32)
    return pos, vel


def step_random_noise(pos: np.ndarray, vel: np.ndarray, width: int, height: int, rng: np.random.Generator) -> None:
    vel[:, 0] += rng.normal(0.0, 0.05, size=vel.shape[0]).astype(np.float32)
    vel[:, 1] += rng.normal(0.0, 0.03, size=vel.shape[0]).astype(np.float32)
    vel[:, 0] = np.clip(vel[:, 0], -1.2, 0.6)
    vel[:, 1] = np.clip(vel[:, 1], -0.6, 0.6)
    pos += vel
    pos[:, 0] %= width
    pos[:, 1] %= height


def render_challenge_random_bg(
    poses: torch.Tensor,
    trans: torch.Tensor,
    betas: torch.Tensor,
    output_path: Path,
    seed: int,
) -> None:
    T = poses.shape[0]
    model = smplx.create(str(MODEL_ROOT), model_type="smpl", gender="neutral", num_betas=10, use_pca=False, batch_size=T)
    with torch.no_grad():
        output = model(global_orient=poses[:, :3], body_pose=poses[:, 3:], betas=betas.expand(T, -1), transl=trans)
    joints = output.joints[:, JOINT_INDICES_17, :]
    pts_2d = joints[..., [0, 2]]

    noise_pos, noise_vel = init_random_noise(CHALLENGE_WIDTH, CHALLENGE_HEIGHT, BACKGROUND_POINT_COUNT, seed=seed)
    rng = np.random.default_rng(seed + 500)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CHALLENGE_WIDTH, CHALLENGE_HEIGHT))

    for t in range(T):
        img = np.zeros((CHALLENGE_HEIGHT, CHALLENGE_WIDTH, 3), dtype=np.uint8)
        step_random_noise(noise_pos, noise_vel, CHALLENGE_WIDTH, CHALLENGE_HEIGHT, rng)
        for x, y in noise_pos:
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)

        for pt in pts_2d[t].cpu().numpy():
            px = int(pt[0] * CHALLENGE_DRAW_SCALE + CHALLENGE_WIDTH // 2)
            py = int(-pt[1] * CHALLENGE_DRAW_SCALE + CHALLENGE_OFFSET_Y)
            cv2.circle(img, (px, py), 3, (255, 255, 255), -1)
        out.write(img)
    out.release()
    convert_to_h264(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate L3 dataset with weak-random background noise.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-path", type=str, default=str(SELECTION_PATH))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-options", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)

    selection_path = Path(args.selection_path)
    with selection_path.open("r", encoding="utf-8") as f:
        selection = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    generated_items = []
    for action, items in selection.items():
        for idx, item in enumerate(items):
            item_id = sanitize_name(f"{action}_{Path(item['local_relpath']).stem}_{idx:02d}")
            challenge_name = f"L3_{item_id}_Challenge.mp4"
            option_name = f"L3_{item_id}_Option.mp4"
            challenge_path = LIB_DIR / challenge_name
            option_path = LIB_DIR / option_name

            poses, trans, betas = load_segment(Path(item["local_path"]), item["start_t"], item["end_t"])
            if args.overwrite or not challenge_path.exists():
                render_challenge_random_bg(poses, trans, betas, challenge_path, seed=args.seed + idx)
            if (not args.skip_options) and (args.overwrite or not option_path.exists()):
                render_option(poses, trans, betas, option_path)

            generated_items.append(
                {
                    "id": item_id,
                    "category": action,
                    "challenge_video": challenge_name,
                    "option_video": None if args.skip_options else option_name,
                    "local_relpath": item["local_relpath"],
                    "start_t": item["start_t"],
                    "end_t": item["end_t"],
                }
            )
            print(f"Generated {item_id}")

    items_by_cat = defaultdict(list)
    for item in generated_items:
        items_by_cat[item["category"]].append(item)

    final_metadata = []
    categories = list(items_by_cat.keys())
    for item in generated_items:
        distractor_cats = rng.sample([c for c in categories if c != item["category"]], 5)
        distractors = []
        for cat in distractor_cats:
            picked = rng.choice(items_by_cat[cat])
            distractors.append({"id": picked["id"], "category": cat, "option_video": picked["option_video"]})
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
                "background_mode": "perturbed_drift",
            }
        )

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
