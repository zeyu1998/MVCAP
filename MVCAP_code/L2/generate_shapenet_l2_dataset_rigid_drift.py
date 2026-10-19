import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh

from .config import (
    BG_COVERAGE,
    BG_SPEED,
    CAMERA_DISTANCE,
    DEFAULT_NUM_SAMPLES,
    DOT_RADIUS,
    FG_COVERAGE,
    FG_SPEED,
    FPS,
    METADATA_JSON,
    SEED,
    SELECTION_JSON,
    THUMB_ROOT,
    TOTAL_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_ROOT,
    VIDEO_TARGET_OCCUPANCY,
    VIDEO_WIDTH,
)
from .shapenet_utils import load_normalized_scene, load_selection, scene_to_single_mesh
from .shapenet_utils import build_three_quarter_camera_pose


def generate_dot_texture(height: int, width: int, coverage: float, radius: int) -> np.ndarray:
    texture = np.zeros((height, width), dtype=np.uint8)
    dot_area = np.pi * (radius ** 2)
    num_dots = int((height * width * coverage) / dot_area)
    xs = np.random.randint(0, width, size=num_dots)
    ys = np.random.randint(0, height, size=num_dots)
    for x, y in zip(xs, ys):
        cv2.circle(texture, (int(x), int(y)), radius, 255, -1)
    return texture


def convert_to_h264(video_path: Path) -> None:
    temp_path = video_path.with_name(video_path.stem + "_h264.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-g",
        "1",
        "-keyint_min",
        "1",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-loglevel",
        "error",
        str(temp_path),
    ]
    subprocess.run(cmd, check=True)
    os.replace(temp_path, video_path)


def rotation_pose(angle_radians: float, scale: float = 1.0) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle_radians) * scale, 0, np.sin(angle_radians) * scale, 0],
            [0, scale, 0, 0],
            [-np.sin(angle_radians) * scale, 0, np.cos(angle_radians) * scale, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def mask_max_extent(mask: np.ndarray) -> int:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0
    return int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def estimate_video_scale(renderer, scene, mesh_node, sample_count: int = 12) -> float:
    sampled_max_extent = 0
    for sample_idx in range(sample_count):
        angle = 2.0 * np.pi * sample_idx / sample_count
        scene.set_pose(mesh_node, pose=rotation_pose(angle))
        color, _ = renderer.render(scene)
        sampled_max_extent = max(sampled_max_extent, mask_max_extent(color[:, :, 0] > 128))

    if sampled_max_extent <= 0:
        return 1.0

    target_extent = int(min(VIDEO_WIDTH, VIDEO_HEIGHT) * VIDEO_TARGET_OCCUPANCY)
    scale = target_extent / sampled_max_extent
    return float(np.clip(scale, 0.55, 2.2))


def generate_video(
    obj_path: Path,
    output_path: Path,
    fg_coverage: float,
    bg_coverage: float,
    fg_speed: int,
    bg_speed: int,
    radius: int,
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
    cam_pose = build_three_quarter_camera_pose(CAMERA_DISTANCE)
    scene.add(cam, pose=cam_pose)

    mesh_scale = estimate_video_scale(renderer, scene, mesh_node)

    tex_bg = generate_dot_texture(VIDEO_HEIGHT, VIDEO_WIDTH, bg_coverage, radius)
    tex_fg = generate_dot_texture(VIDEO_HEIGHT, VIDEO_WIDTH, fg_coverage, radius)

    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
    out = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=False)

    degrees_per_frame = 360.0 / TOTAL_FRAMES
    for frame_idx in range(TOTAL_FRAMES):
        angle = np.radians(frame_idx * degrees_per_frame)
        rot = rotation_pose(angle, scale=mesh_scale)
        scene.set_pose(mesh_node, pose=rot)
        color, _ = renderer.render(scene)
        mask = color[:, :, 0] > 128

        curr_bg = np.roll(tex_bg, frame_idx * bg_speed, axis=1)
        curr_fg = np.roll(tex_fg, frame_idx * fg_speed, axis=1)
        frame = curr_bg.copy()
        frame[mask] = curr_fg[mask]
        out.write(frame)

    out.release()
    renderer.delete()
    convert_to_h264(raw_path)
    os.replace(raw_path, output_path)


def generate_video_with_retry(
    obj_path: Path,
    output_path: Path,
    fg_coverage: float,
    bg_coverage: float,
    fg_speed: int,
    bg_speed: int,
    radius: int,
    retries: int = 5,
    retry_delay_s: float = 1.5,
) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            generate_video(
                obj_path,
                output_path,
                fg_coverage=fg_coverage,
                bg_coverage=bg_coverage,
                fg_speed=fg_speed,
                bg_speed=bg_speed,
                radius=radius,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
            print(
                f"[retry {attempt}/{retries}] {output_path.name} failed with {type(exc).__name__}: {exc}. "
                f"Retrying in {retry_delay_s:.1f}s...",
                flush=True,
            )
            time.sleep(retry_delay_s)
    if last_error is not None:
        raise last_error


def make_balanced_targets(selection: dict, num_samples: int, rng: random.Random) -> list[tuple[str, dict]]:
    categories = list(selection.keys())
    pool = (categories * ((num_samples // len(categories)) + 1))[:num_samples]
    rng.shuffle(pool)

    used_ids = set()
    targets = []
    for category in pool:
        choices = [item for item in selection[category] if item["model_id"] not in used_ids]
        if not choices:
            choices = selection[category]
        item = rng.choice(choices)
        used_ids.add(item["model_id"])
        targets.append((category, item))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate L2 ShapeNetCore challenges with rigid-drift background.")
    parser.add_argument("--selection-path", type=str, default=str(SELECTION_JSON))
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fg-coverage", type=float, default=FG_COVERAGE)
    parser.add_argument("--bg-coverage", type=float, default=BG_COVERAGE)
    parser.add_argument("--fg-speed", type=int, default=FG_SPEED)
    parser.add_argument("--bg-speed", type=int, default=BG_SPEED)
    parser.add_argument("--dot-radius", type=int, default=DOT_RADIUS)
    args = parser.parse_args()
    np.random.seed(args.seed)

    selection = load_selection(Path(args.selection_path))
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    targets = make_balanced_targets(selection, args.num_samples, rng)

    final_metadata = []
    categories = list(selection.keys())
    for idx, (target_category, target_item) in enumerate(targets):
        target_name = target_item.get("sample_name", f"{target_category}_{idx:02d}")
        challenge_id = target_name
        video_path = VIDEO_ROOT / f"{challenge_id}.mp4"

        if args.overwrite or not video_path.exists():
            print(f"[{idx + 1}/{len(targets)}] Generating {challenge_id} from {target_category}/{target_item['model_id']}")
            generate_video_with_retry(
                Path(target_item["obj_path"]),
                video_path,
                fg_coverage=args.fg_coverage,
                bg_coverage=args.bg_coverage,
                fg_speed=args.fg_speed,
                bg_speed=args.bg_speed,
                radius=args.dot_radius,
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
                "id": challenge_id,
                "video_path": str(video_path.relative_to(Path.cwd())),
                "target": {
                    "category": target_category,
                    "model_id": target_item["model_id"],
                    "sample_name": target_name,
                    "thumbnail": str((THUMB_ROOT / target_category / f"{target_name}.png").relative_to(Path.cwd())),
                    "obj_path": target_item["obj_path"],
                },
                "distractors": distractors,
                "background_mode": "rigid_drift",
            }
        )

    METADATA_JSON.write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {METADATA_JSON}")


if __name__ == "__main__":
    main()
