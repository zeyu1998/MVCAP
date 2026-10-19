import argparse
import inspect
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyrender
import smplx
import torch
import torch.nn.functional as F
import trimesh


if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
if "int" not in np.__dict__:
    np.int = int
if "bool" not in np.__dict__:
    np.bool = bool
if "float" not in np.__dict__:
    np.float = float
if "complex" not in np.__dict__:
    np.complex = complex
if "object" not in np.__dict__:
    np.object = object
if "unicode" not in np.__dict__:
    np.unicode = str
if "str" not in np.__dict__:
    np.str = str


ROOT_DIR = Path(__file__).resolve().parents[2]
SELECTION_PATH = ROOT_DIR / "dataset" / "L3_babel" / "l3_14class_selection.json"
OUTPUT_DIR = ROOT_DIR / "dataset" / "L3_selected_generated_rigid_drift"
LIB_DIR = OUTPUT_DIR / "library"
METADATA_PATH = OUTPUT_DIR / "L3_metadata.json"
MODEL_ROOT = Path(os.environ.get("MVCAP_SMPL_ROOT", ROOT_DIR / "assets" / "body_models")).expanduser()

FPS = 30
DURATION = 3.0
TARGET_FRAMES = int(FPS * DURATION)
CHALLENGE_WIDTH = 448
CHALLENGE_HEIGHT = 448
OPTION_WIDTH = 256
OPTION_HEIGHT = 256
CHALLENGE_DRAW_SCALE = 250.0
CHALLENGE_OFFSET_Y = CHALLENGE_HEIGHT * 0.72
BACKGROUND_POINT_COUNT = 170
OPTION_TARGET_HEIGHT_RATIO = 0.65
OPTION_TARGET_WIDTH_RATIO = 0.8
# 17-point biological motion: head, neck, upper limbs, pelvis/hips, knees, ankles, feet
JOINT_INDICES_17 = [15, 12, 16, 17, 18, 19, 20, 21, 0, 1, 2, 4, 5, 7, 8, 10, 11]


def sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def convert_to_h264(video_path: Path, crop: str | None = None) -> None:
    temp_path = video_path.with_name(video_path.stem + "_temp.mp4")
    cmd = ["ffmpeg", "-i", str(video_path)]
    if crop:
        cmd.extend(["-vf", f"crop={crop}"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
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
            str(temp_path),
        ]
    )
    print(f"[ffmpeg] start {video_path.name}" + (f" crop={crop}" if crop else ""), flush=True)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    os.replace(temp_path, video_path)
    print(f"[ffmpeg] done  {video_path.name}", flush=True)


def load_segment(npz_path: Path, start_t: float, end_t: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(npz_path)
    poses_raw = torch.tensor(data["poses"], dtype=torch.float32)
    poses = poses_raw[:, :72]
    betas = torch.tensor(data["betas"][:10], dtype=torch.float32).unsqueeze(0)
    trans = torch.tensor(data["trans"], dtype=torch.float32)

    src_fps = float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 120.0
    total_frames = poses.shape[0]
    clip_duration = total_frames / src_fps if src_fps > 0 else 0.0
    start_t = max(0.0, min(float(start_t), clip_duration))
    end_t = max(start_t, min(float(end_t), clip_duration))
    start_idx = max(0, int(np.floor(start_t * src_fps)))
    end_idx = max(start_idx + 1, int(np.ceil(end_t * src_fps)))
    start_idx = min(start_idx, total_frames)
    end_idx = min(end_idx, total_frames)
    poses = poses[start_idx:end_idx]
    trans = trans[start_idx:end_idx]

    if poses.shape[0] == 0:
        raise ValueError(
            f"Empty segment after clipping bounds for {npz_path}: "
            f"start_t={start_t:.3f}, end_t={end_t:.3f}, total_frames={total_frames}, fps={src_fps}"
        )

    stride = max(1, int(src_fps // FPS))
    poses = poses[::stride]
    trans = trans[::stride]

    if poses.shape[0] < TARGET_FRAMES:
        # Time-resample short clips to 3 seconds instead of hard looping them.
        # This avoids visible discontinuities at the loop seam when scrubbing.
        def resize_time(seq: torch.Tensor, target_frames: int) -> torch.Tensor:
            if seq.shape[0] == 1:
                return seq.repeat(target_frames, 1)
            seq_1d = seq.transpose(0, 1).unsqueeze(0)
            resized = F.interpolate(seq_1d, size=target_frames, mode="linear", align_corners=True)
            return resized.squeeze(0).transpose(0, 1).contiguous()

        poses = resize_time(poses, TARGET_FRAMES)
        trans = resize_time(trans, TARGET_FRAMES)

    if poses.shape[0] > TARGET_FRAMES:
        start = (poses.shape[0] - TARGET_FRAMES) // 2
        poses = poses[start:start + TARGET_FRAMES]
        trans = trans[start:start + TARGET_FRAMES]

    trans[:, 0] = 0
    trans[:, 2] = 0
    return poses, trans, betas


def render_challenge(poses: torch.Tensor, trans: torch.Tensor, betas: torch.Tensor, output_path: Path) -> None:
    T = poses.shape[0]
    model = smplx.create(str(MODEL_ROOT), model_type="smpl", gender="neutral", num_betas=10, use_pca=False, batch_size=T)
    with torch.no_grad():
        output = model(global_orient=poses[:, :3], body_pose=poses[:, 3:], betas=betas.expand(T, -1), transl=trans)
    joints = output.joints[:, JOINT_INDICES_17, :]
    pts_2d = joints[..., [0, 2]]

    noise = [[np.random.randint(0, CHALLENGE_WIDTH), np.random.randint(0, CHALLENGE_HEIGHT)] for _ in range(BACKGROUND_POINT_COUNT)]
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CHALLENGE_WIDTH, CHALLENGE_HEIGHT))

    for t in range(T):
        img = np.zeros((CHALLENGE_HEIGHT, CHALLENGE_WIDTH, 3), dtype=np.uint8)
        for p in noise:
            p[0] -= 4
            if p[0] < 0:
                p[0] += CHALLENGE_WIDTH
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)
        for pt in pts_2d[t].cpu().numpy():
            px = int(pt[0] * CHALLENGE_DRAW_SCALE + CHALLENGE_WIDTH // 2)
            py = int(-pt[1] * CHALLENGE_DRAW_SCALE + CHALLENGE_OFFSET_Y)
            cv2.circle(img, (px, py), 3, (255, 255, 255), -1)
        out.write(img)
    out.release()
    convert_to_h264(output_path)


def render_option(poses: torch.Tensor, trans: torch.Tensor, betas: torch.Tensor, output_path: Path) -> None:
    T = poses.shape[0]
    model = smplx.create(str(MODEL_ROOT), model_type="smpl", gender="neutral", num_betas=10, use_pca=False, batch_size=T)
    with torch.no_grad():
        output = model(global_orient=poses[:, :3], body_pose=poses[:, 3:], betas=betas.expand(T, -1), transl=trans)
    vertices = output.vertices.numpy()
    faces = model.faces
    renderer = pyrender.OffscreenRenderer(OPTION_WIDTH, OPTION_HEIGHT)
    base_tf = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=OPTION_WIDTH / OPTION_HEIGHT)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0, 0.85, 3.3]
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.8, 0.6, 0.5, 1.0], metallicFactor=0.0, roughnessFactor=0.5)
    rendered_frames: list[np.ndarray] = []
    global_x1 = OPTION_WIDTH
    global_y1 = OPTION_HEIGHT
    global_x2 = 0
    global_y2 = 0

    for t in range(T):
        if t == 0 or (t + 1) % 15 == 0 or t == T - 1:
            print(f"[option] {output_path.name} frame {t + 1}/{T}", flush=True)
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
        mesh = trimesh.Trimesh(vertices=vertices[t], faces=faces)
        mesh.apply_transform(base_tf)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        color, depth = renderer.render(scene)
        frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        rendered_frames.append(frame)
        mask = depth > 0
        if mask.any():
            ys, xs = np.where(mask)
            global_x1 = min(global_x1, int(xs.min()))
            global_y1 = min(global_y1, int(ys.min()))
            global_x2 = max(global_x2, int(xs.max()) + 1)
            global_y2 = max(global_y2, int(ys.max()) + 1)

    renderer.delete()
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OPTION_WIDTH, OPTION_HEIGHT))

    if global_x2 <= global_x1 or global_y2 <= global_y1:
        global_x1, global_y1, global_x2, global_y2 = 0, 0, OPTION_WIDTH, OPTION_HEIGHT

    crop_h = global_y2 - global_y1
    crop_w = global_x2 - global_x1
    target_h = int(round(OPTION_HEIGHT * OPTION_TARGET_HEIGHT_RATIO))
    target_w = int(round(OPTION_WIDTH * OPTION_TARGET_WIDTH_RATIO))
    scale = min(target_h / max(crop_h, 1), target_w / max(crop_w, 1))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    paste_x = (OPTION_WIDTH - resized_w) // 2
    paste_y = (OPTION_HEIGHT - resized_h) // 2

    for frame in rendered_frames:
        cropped = frame[global_y1:global_y2, global_x1:global_x2]
        resized = cv2.resize(cropped, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((OPTION_HEIGHT, OPTION_WIDTH, 3), dtype=np.uint8)
        canvas[paste_y : paste_y + resized_h, paste_x : paste_x + resized_w] = resized
        out.write(canvas)

    out.release()
    convert_to_h264(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate L3 dataset from balanced BABEL segment selection.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dot fields and distractor assignment.")
    parser.add_argument(
        "--selection-path",
        type=str,
        default=str(SELECTION_PATH),
        help="Path to the JSON selection file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for generated videos and metadata.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing videos.")
    args = parser.parse_args()
    np.random.seed(args.seed)

    selection_path = Path(args.selection_path)
    output_dir = Path(args.output_dir)
    lib_dir = output_dir / "library"
    metadata_path = output_dir / "L3_metadata.json"
    with selection_path.open("r", encoding="utf-8") as f:
        selection = json.load(f)

    lib_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    generated_items = []
    for action, items in selection.items():
        for idx, item in enumerate(items):
            item_id = sanitize_name(f"{action}_{Path(item['local_relpath']).stem}_{idx:02d}")
            challenge_name = f"L3_{item_id}_Challenge.mp4"
            option_name = f"L3_{item_id}_Option.mp4"
            challenge_path = lib_dir / challenge_name
            option_path = lib_dir / option_name

            print(f"[item] start {item_id}", flush=True)
            poses, trans, betas = load_segment(Path(item["local_path"]), item["start_t"], item["end_t"])
            if args.overwrite or not challenge_path.exists():
                tic = time.time()
                print(f"[item] challenge {item_id}", flush=True)
                render_challenge(poses, trans, betas, challenge_path)
                print(f"[item] challenge done {item_id} {time.time() - tic:.1f}s", flush=True)
            if args.overwrite or not option_path.exists():
                tic = time.time()
                print(f"[item] option {item_id}", flush=True)
                render_option(poses, trans, betas, option_path)
                print(f"[item] option done {item_id} {time.time() - tic:.1f}s", flush=True)

            generated_items.append(
                {
                    "id": item_id,
                    "category": action,
                    "challenge_video": challenge_name,
                    "option_video": option_name,
                    "local_relpath": item["local_relpath"],
                    "start_t": item["start_t"],
                    "end_t": item["end_t"],
                }
            )
            print(f"Generated {item_id}", flush=True)

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
                "background_mode": "rigid_drift",
            }
        )

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
