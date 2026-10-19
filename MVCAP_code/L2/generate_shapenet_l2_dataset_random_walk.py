import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pyrender

from .config import (
    BG_COVERAGE,
    CAMERA_DISTANCE,
    DEFAULT_NUM_SAMPLES,
    DOT_RADIUS,
    FG_COVERAGE,
    FG_SPEED,
    SEED,
    SELECTION_JSON,
    THUMB_ROOT,
    TOTAL_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_TARGET_OCCUPANCY,
    VIDEO_WIDTH,
)
from .generate_shapenet_l2_dataset_rigid_drift import (
    convert_to_h264,
    estimate_video_scale,
    generate_dot_texture,
    make_balanced_targets,
    rotation_pose,
)
from .shapenet_utils import build_three_quarter_camera_pose, load_normalized_scene, load_selection, scene_to_single_mesh


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT_DIR / "dataset" / "L2_shapenet_random_walk"
VIDEO_ROOT = OUTPUT_ROOT / "videos"
METADATA_JSON = OUTPUT_ROOT / "L2_metadata.json"


def init_random_walk_dots(width: int, height: int, coverage: float, radius: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    dot_area = np.pi * (radius**2)
    num_dots = int((width * height * coverage) / dot_area)
    rng = np.random.default_rng(seed)
    pos = np.stack(
        [
            rng.uniform(0, width, size=num_dots),
            rng.uniform(0, height, size=num_dots),
        ],
        axis=1,
    ).astype(np.float32)
    vel = np.stack(
        [
            rng.normal(0.0, 0.48, size=num_dots),
            rng.normal(0.0, 0.28, size=num_dots),
        ],
        axis=1,
    ).astype(np.float32)
    return pos, vel


def step_random_walk_dots(pos: np.ndarray, vel: np.ndarray, width: int, height: int, rng: np.random.Generator) -> None:
    vel[:, 0] += rng.normal(0.0, 0.16, size=vel.shape[0]).astype(np.float32)
    vel[:, 1] += rng.normal(0.0, 0.12, size=vel.shape[0]).astype(np.float32)
    vel[:, 0] = np.clip(vel[:, 0], -2.0, 2.0)
    vel[:, 1] = np.clip(vel[:, 1], -1.5, 1.5)
    pos += vel
    pos[:, 0] %= width
    pos[:, 1] %= height


def dots_to_frame(pos: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    frame = np.zeros((height, width), dtype=np.uint8)
    for x, y in pos:
        cv2.circle(frame, (int(x), int(y)), radius, 255, -1)
    return frame


def generate_video_random_walk_bg(
    obj_path: Path,
    output_path: Path,
    fg_coverage: float,
    bg_coverage: float,
    fg_speed: int,
    radius: int,
    seed: int,
) -> None:
    scene_tm = load_normalized_scene(obj_path)
    mesh = scene_to_single_mesh(scene_tm)

    renderer = pyrender.OffscreenRenderer(VIDEO_WIDTH, VIDEO_HEIGHT)
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0, 0])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        emissiveFactor=[1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=VIDEO_WIDTH / VIDEO_HEIGHT)
    scene.add(cam, pose=build_three_quarter_camera_pose(CAMERA_DISTANCE))

    mesh_scale = estimate_video_scale(renderer, scene, mesh_node)
    tex_fg = generate_dot_texture(VIDEO_HEIGHT, VIDEO_WIDTH, fg_coverage, radius)
    bg_pos, bg_vel = init_random_walk_dots(VIDEO_WIDTH, VIDEO_HEIGHT, bg_coverage, radius, seed=seed)
    rng = np.random.default_rng(seed + 2301)

    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=False)

    degrees_per_frame = 360.0 / TOTAL_FRAMES
    for frame_idx in range(TOTAL_FRAMES):
        angle = np.radians(frame_idx * degrees_per_frame)
        scene.set_pose(mesh_node, pose=rotation_pose(angle, scale=mesh_scale))
        color, _ = renderer.render(scene)
        mask = color[:, :, 0] > 128

        step_random_walk_dots(bg_pos, bg_vel, VIDEO_WIDTH, VIDEO_HEIGHT, rng)
        curr_bg = dots_to_frame(bg_pos, VIDEO_WIDTH, VIDEO_HEIGHT, radius)
        curr_fg = np.roll(tex_fg, int(frame_idx * fg_speed), axis=1)

        frame = curr_bg.copy()
        frame[mask] = curr_fg[mask]
        writer.write(frame)

    writer.release()
    renderer.delete()
    convert_to_h264(raw_path)
    os.replace(raw_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate L2 ShapeNetCore dataset with pure random-walk background.")
    parser.add_argument("--selection-path", type=str, default=str(SELECTION_JSON))
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fg-coverage", type=float, default=FG_COVERAGE)
    parser.add_argument("--bg-coverage", type=float, default=BG_COVERAGE)
    parser.add_argument("--fg-speed", type=int, default=FG_SPEED)
    parser.add_argument("--dot-radius", type=int, default=DOT_RADIUS)
    args = parser.parse_args()
    np.random.seed(args.seed)

    selection = load_selection(Path(args.selection_path))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    targets = make_balanced_targets(selection, args.num_samples, rng)

    final_metadata = []
    categories = list(selection.keys())
    for idx, (target_category, target_item) in enumerate(targets):
        target_name = target_item.get("sample_name", f"{target_category}_{idx:02d}")
        video_path = VIDEO_ROOT / f"{target_name}.mp4"

        if args.overwrite or not video_path.exists():
            print(f"[{idx + 1}/{len(targets)}] Generating random-walk {target_name}")
            generate_video_random_walk_bg(
                Path(target_item["obj_path"]),
                video_path,
                fg_coverage=args.fg_coverage,
                bg_coverage=args.bg_coverage,
                fg_speed=args.fg_speed,
                radius=args.dot_radius,
                seed=args.seed + idx,
            )

        distractor_categories = rng.sample([cat for cat in categories if cat != target_category], 5)
        distractors = []
        for distractor_category in distractor_categories:
            distractor_item = rng.choice(selection[distractor_category])
            distractor_name = distractor_item.get("sample_name", f"{distractor_category}_xx")
            distractors.append(
                {
                    "category": distractor_category,
                    "model_id": distractor_item["model_id"],
                    "sample_name": distractor_name,
                    "thumbnail": str((THUMB_ROOT / distractor_category / f"{distractor_name}.png").relative_to(Path.cwd())),
                }
            )

        final_metadata.append(
            {
                "id": target_name,
                "video_path": str(video_path.relative_to(Path.cwd())),
                "target": {
                    "category": target_category,
                    "model_id": target_item["model_id"],
                    "sample_name": target_name,
                    "thumbnail": str((THUMB_ROOT / target_category / f"{target_name}.png").relative_to(Path.cwd())),
                    "obj_path": target_item["obj_path"],
                },
                "distractors": distractors,
                "background_mode": "random_walk",
            }
        )

    METADATA_JSON.write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {METADATA_JSON}")


if __name__ == "__main__":
    main()
