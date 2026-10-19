import argparse
import json
import shutil
import subprocess
from pathlib import Path


def reencode_all_intra(src_video: Path, dst_video: Path, crf: int) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(src_video),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-g",
        "1",
        "-keyint_min",
        "1",
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-an",
        "-y",
        str(dst_video),
    ]
    subprocess.run(cmd, check=True)


def copy_json_files(src_dir: Path, dst_dir: Path) -> None:
    for path in src_dir.glob("*.json"):
        dst_path = dst_dir / path.name
        dst_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an all-intra L3 dataset copy for slider-friendly playback.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Source L3 dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination L3 dataset directory.")
    parser.add_argument("--crf", type=int, default=18, help="CRF to use for all-intra re-encoding.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    src_library = input_dir / "library"
    dst_library = output_dir / "library"

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_library.mkdir(parents=True, exist_ok=True)

    copy_json_files(input_dir, output_dir)

    videos = sorted(src_library.glob("*.mp4"))
    print(f"Re-encoding {len(videos)} videos from {src_library} -> {dst_library}")
    for idx, src_video in enumerate(videos, start=1):
        dst_video = dst_library / src_video.name
        reencode_all_intra(src_video, dst_video, args.crf)
        print(f"[{idx}/{len(videos)}] {src_video.name}")

    summary = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "video_count": len(videos),
        "encoding": "h264_all_intra",
        "crf": args.crf,
    }
    (output_dir / "all_intra_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
