"""Generate the seven-letter README illustration without changing the benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MVCAP_code.L1.config import FG_SPEED
from MVCAP_code.L1.generate_l1_dataset_rigid_drift import L1TextCaptchaGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = L1TextCaptchaGenerator()
    text = "CAPTCHA"
    main_path = args.output_dir / "l1-captcha-main.mp4"
    fg_path = args.output_dir / "l1-captcha-fg.mp4"
    answer_path = args.output_dir / "l1-captcha-dots.png"
    semantic_path = args.output_dir / "l1-semantic-target.png"

    np.random.seed(args.seed)
    generator.generate_single(text, main_path, answer_path)

    # Replay the main generator's two texture draws, retaining the same foreground.
    np.random.seed(args.seed)
    generator.generate_dot_texture(generator.height, generator.width)
    foreground = generator.generate_dot_texture(generator.height, generator.width)
    mask = generator.build_text_mask(text)
    first_frame = np.zeros_like(foreground)
    first_frame[mask] = foreground[mask]
    assert np.array_equal(first_frame, cv2.imread(str(answer_path), cv2.IMREAD_GRAYSCALE))

    writer = cv2.VideoWriter(
        str(fg_path), cv2.VideoWriter_fourcc(*"mp4v"), generator.fps,
        (generator.width, generator.height), isColor=False,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {fg_path}")
    for index in range(generator.total_frames):
        frame = np.zeros_like(foreground)
        shifted = np.roll(foreground, int(index * FG_SPEED), axis=1)
        frame[mask] = shifted[mask]
        writer.write(frame)
    writer.release()
    generator.convert_to_h264(fg_path)
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(semantic_path)

    metadata = {
        "id": "L1_CAPTCHA_showcase", "text": text, "seed": args.seed,
        "illustration_only": True, "in_frozen_benchmark": False,
        "note": "Seven-letter illustration; standard MVCAP-Bench L1 uses four characters.",
        "main": main_path.name, "foreground_only": fg_path.name,
        "semantic_target": semantic_path.name,
        "foreground_texture_identical_between_conditions": True,
        "fps": generator.fps, "frames": generator.total_frames,
        "font": str(generator.find_font(100).getname()),
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (main_path, fg_path, semantic_path)
        },
    }
    (args.output_dir / "l1-showcase.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
