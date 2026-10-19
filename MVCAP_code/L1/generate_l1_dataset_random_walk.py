import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import (
    BG_SPEED,
    DEFAULT_NUM_SAMPLES,
    DOT_COVERAGE,
    DOT_RADIUS,
    FG_SPEED,
    SEED,
    TEXT_LENGTH,
)
from .generate_l1_dataset_rigid_drift import L1TextCaptchaGenerator, sample_text


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT_DIR / "dataset" / "L1_text_dataset_random_walk"
METADATA_JSON = DATASET_ROOT / "dataset_metadata.json"


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
            rng.normal(0.0, 0.55, size=num_dots),
            rng.normal(0.0, 0.35, size=num_dots),
        ],
        axis=1,
    ).astype(np.float32)
    return pos, vel


def step_random_walk_dots(pos: np.ndarray, vel: np.ndarray, width: int, height: int, rng: np.random.Generator) -> None:
    vel[:, 0] += rng.normal(0.0, 0.22, size=vel.shape[0]).astype(np.float32)
    vel[:, 1] += rng.normal(0.0, 0.18, size=vel.shape[0]).astype(np.float32)
    vel[:, 0] = np.clip(vel[:, 0], -2.4, 2.4)
    vel[:, 1] = np.clip(vel[:, 1], -2.0, 2.0)
    pos += vel
    pos[:, 0] %= width
    pos[:, 1] %= height


def dots_to_frame(pos: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    frame = np.zeros((height, width), dtype=np.uint8)
    for x, y in pos:
        cv2.circle(frame, (int(x), int(y)), radius, 255, -1)
    return frame


def convert_to_h264(video_path: Path) -> None:
    temp_path = video_path.with_name(video_path.stem + "_temp.mp4")
    os.replace(video_path, temp_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_path),
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
        str(video_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        temp_path.unlink()
    except Exception:
        if temp_path.exists():
            os.replace(temp_path, video_path)
        raise


def generate_single_random_walk_bg(
    generator: L1TextCaptchaGenerator,
    text: str,
    video_path: Path,
    png_path: Path,
    fg_speed: int,
    bg_seed: int,
) -> None:
    text_mask = generator.build_text_mask(text)
    tex_fg = generator.generate_dot_texture(generator.height, generator.width)

    answer_canvas = np.zeros((generator.height, generator.width), dtype=np.uint8)
    answer_canvas[text_mask] = tex_fg[text_mask]
    cv2.imwrite(str(png_path), answer_canvas)

    bg_pos, bg_vel = init_random_walk_dots(generator.width, generator.height, DOT_COVERAGE, DOT_RADIUS, seed=bg_seed)
    rng = np.random.default_rng(bg_seed + 1701)

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), generator.fps, (generator.width, generator.height), isColor=False)
    for frame_idx in range(generator.total_frames):
        step_random_walk_dots(bg_pos, bg_vel, generator.width, generator.height, rng)
        curr_bg = dots_to_frame(bg_pos, generator.width, generator.height, DOT_RADIUS)
        curr_fg = np.roll(tex_fg, int(frame_idx * fg_speed), axis=1)

        frame = curr_bg.copy()
        frame[text_mask] = curr_fg[text_mask]
        writer.write(frame)
    writer.release()
    convert_to_h264(video_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate L1 dataset with pure random-walk background motion.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fg-speed", type=int, default=FG_SPEED)
    parser.add_argument("--bg-speed", type=int, default=BG_SPEED)
    args = parser.parse_args()
    np.random.seed(args.seed)

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    generator = L1TextCaptchaGenerator()

    metadata = []
    for index in range(args.num_samples):
        sample_name = f"L1RW_{index:03d}"
        video_path = DATASET_ROOT / f"{sample_name}.mp4"
        png_path = DATASET_ROOT / f"{sample_name}.png"
        text = sample_text(rng, length=TEXT_LENGTH)

        if args.overwrite or not video_path.exists() or not png_path.exists():
            generate_single_random_walk_bg(
                generator,
                text,
                video_path,
                png_path,
                fg_speed=args.fg_speed,
                bg_seed=args.seed + index,
            )

        metadata.append(
            {
                "id": sample_name,
                "text": text,
                "video_path": str(video_path.relative_to(Path.cwd())),
                "answer_png": str(png_path.relative_to(Path.cwd())),
                "background_mode": "random_walk",
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == args.num_samples:
            print(f"Generated {index + 1}/{args.num_samples}")

    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {METADATA_JSON}")


if __name__ == "__main__":
    main()
