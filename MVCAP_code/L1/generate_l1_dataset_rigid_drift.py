import argparse
import json
import os
import random
import string
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    ALPHABET,
    BG_SPEED,
    DATASET_ROOT,
    DEFAULT_NUM_SAMPLES,
    DOT_COVERAGE,
    DOT_RADIUS,
    DURATION,
    FG_SPEED,
    FPS,
    METADATA_JSON,
    NUM_DISTRACTORS,
    SEED,
    TEXT_LENGTH,
    TOTAL_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


class L1TextCaptchaGenerator:
    def __init__(self, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT, fps: int = FPS, duration: float = DURATION):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.total_frames = TOTAL_FRAMES

    def find_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        possible_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def generate_dot_texture(self, height: int, width: int, coverage: float = DOT_COVERAGE, radius: int = DOT_RADIUS) -> np.ndarray:
        texture = np.zeros((height, width), dtype=np.uint8)
        dot_area = np.pi * (radius**2)
        num_dots = int((height * width * coverage) / dot_area)
        xs = np.random.randint(0, width, size=num_dots)
        ys = np.random.randint(0, height, size=num_dots)
        for x, y in zip(xs, ys):
            cv2.circle(texture, (int(x), int(y)), radius, 255, -1)
        return texture

    def build_text_mask(self, text: str) -> np.ndarray:
        canvas = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(canvas)

        fontsize = int(self.height * 0.9)
        font = self.find_font(fontsize)
        while fontsize > 10:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            if text_width < self.width * 0.9 and text_height < self.height * 0.9:
                break
            fontsize -= 5
            font = self.find_font(fontsize)

        draw.text((self.width // 2, self.height // 2), text, fill=255, font=font, anchor="mm")
        return np.array(canvas) > 128

    def convert_to_h264(self, video_path: Path) -> None:
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

    def generate_single(self, text: str, video_path: Path, png_path: Path, bg_speed: int = BG_SPEED, fg_speed: int = FG_SPEED) -> None:
        text_mask = self.build_text_mask(text)
        tex_bg = self.generate_dot_texture(self.height, self.width)
        tex_fg = self.generate_dot_texture(self.height, self.width)

        answer_canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        answer_canvas[text_mask] = tex_fg[text_mask]
        cv2.imwrite(str(png_path), answer_canvas)

        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height), isColor=False)
        for frame_idx in range(self.total_frames):
            curr_bg = np.roll(tex_bg, int(frame_idx * bg_speed), axis=1)
            curr_fg = np.roll(tex_fg, int(frame_idx * fg_speed), axis=1)
            frame = curr_bg.copy()
            frame[text_mask] = curr_fg[text_mask]
            writer.write(frame)
        writer.release()
        self.convert_to_h264(video_path)


def sample_text(rng: random.Random, length: int = TEXT_LENGTH) -> str:
    return "".join(rng.choices(ALPHABET, k=length))


def sample_distractors(rng: random.Random, correct_text: str, count: int = NUM_DISTRACTORS) -> list[str]:
    distractors = set()
    while len(distractors) < count:
        candidate = sample_text(rng)
        if candidate != correct_text:
            distractors.add(candidate)
    return sorted(distractors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate L1 text camouflage CAPTCHA videos.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bg-speed", type=int, default=BG_SPEED)
    parser.add_argument("--fg-speed", type=int, default=FG_SPEED)
    args = parser.parse_args()
    np.random.seed(args.seed)

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    generator = L1TextCaptchaGenerator()

    metadata = []
    for index in range(args.num_samples):
        sample_name = f"L1_{index:03d}"
        video_path = DATASET_ROOT / f"{sample_name}.mp4"
        png_path = DATASET_ROOT / f"{sample_name}.png"
        text = sample_text(rng)
        distractor_texts = sample_distractors(rng, text)

        if args.overwrite or not video_path.exists() or not png_path.exists():
            generator.generate_single(text, video_path, png_path, bg_speed=args.bg_speed, fg_speed=args.fg_speed)

        metadata.append(
            {
                "id": sample_name,
                "text": text,
                "video_path": str(video_path.relative_to(Path.cwd())),
                "answer_png": str(png_path.relative_to(Path.cwd())),
                "target_option": {"id": sample_name, "text": text},
                "distractors": [
                    {"id": f"{sample_name}_d{i}", "text": distractor}
                    for i, distractor in enumerate(distractor_texts)
                ],
                "background_mode": "rigid_drift",
            }
        )

        if (index + 1) % 10 == 0 or index + 1 == args.num_samples:
            print(f"Generated {index + 1}/{args.num_samples}")

    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {METADATA_JSON}")


if __name__ == "__main__":
    main()
