from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MVCAP_code.L1.config import (  # noqa: E402
    BG_SPEED as L1_BG_SPEED,
    DOT_COVERAGE as L1_DOT_COVERAGE,
    DOT_RADIUS as L1_DOT_RADIUS,
    FG_SPEED as L1_FG_SPEED,
    FPS as L1_FPS,
    TEXT_LENGTH as L1_TEXT_LENGTH,
    TOTAL_FRAMES as L1_TOTAL_FRAMES,
    VIDEO_HEIGHT as L1_VIDEO_HEIGHT,
    VIDEO_WIDTH as L1_VIDEO_WIDTH,
)
from MVCAP_code.L1.generate_l1_dataset_rigid_drift import L1TextCaptchaGenerator  # noqa: E402
from MVCAP_code.L2.config import (  # noqa: E402
    BG_COVERAGE as L2_BG_COVERAGE,
    BG_SPEED as L2_BG_SPEED,
    CAMERA_DISTANCE as L2_CAMERA_DISTANCE,
    DOT_RADIUS as L2_DOT_RADIUS,
    FG_COVERAGE as L2_FG_COVERAGE,
    FG_SPEED as L2_FG_SPEED,
    FPS as L2_FPS,
    TOTAL_FRAMES as L2_TOTAL_FRAMES,
    VIDEO_HEIGHT as L2_VIDEO_HEIGHT,
    VIDEO_WIDTH as L2_VIDEO_WIDTH,
)
from MVCAP_code.L2.shapenet_utils import (  # noqa: E402
    build_three_quarter_camera_pose,
    load_normalized_scene,
    scene_to_single_mesh,
)


def preferred_metadata(directory: str, subset_name: str, full_name: str) -> Path:
    """Use a frozen subset when supplied, otherwise use freshly generated data."""
    root = ROOT_DIR / "dataset" / directory
    subset = root / subset_name
    return subset if subset.is_file() else root / full_name


L1_METADATA = preferred_metadata("L1_text_dataset_rigid_drift", "L1_200_metadata.json", "dataset_metadata.json")
L2_METADATA = preferred_metadata("L2_shapenet_rigid_drift", "L2_200_metadata.json", "L2_metadata.json")
L3_METADATA = preferred_metadata("L3_selected_generated_rigid_drift", "L3_200_metadata.json", "L3_metadata.json")
L1_200_METADATA = L1_METADATA
L2_200_METADATA = L2_METADATA
L3_200_METADATA = L3_METADATA


ABLATION_SPECS = {
    "foreground_only": {
        "code": "A",
        "title": "Foreground-Only Ablation",
        "description": "Remove the background carrier and keep only the moving foreground structure.",
    },
}


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def load_first_sample(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))[0]


def load_sample_by_id(path: Path, sample_id: str) -> dict:
    samples = json.loads(path.read_text(encoding="utf-8"))
    for sample in samples:
        if sample.get("id") == sample_id:
            return sample
    raise KeyError(f"Sample id '{sample_id}' not found in {path}")


def load_samples(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_dot_texture(height: int, width: int, coverage: float, radius: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = np.zeros((height, width), dtype=np.uint8)
    dot_area = np.pi * (radius ** 2)
    num_dots = int((height * width * coverage) / dot_area)
    xs = rng.integers(0, width, size=num_dots)
    ys = rng.integers(0, height, size=num_dots)
    for x, y in zip(xs, ys):
        cv2.circle(texture, (int(x), int(y)), radius, 255, -1)
    return texture


def convert_to_all_intra_h264(video_path: Path) -> None:
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
        "-sc_threshold",
        "0",
        "-bf",
        "0",
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


def save_preview(frame: np.ndarray, preview_path: Path) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), frame)


def write_gray_video(frames: list[np.ndarray], output_path: Path, fps: int, save_preview_png: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = frames[0]
    height, width = first_frame.shape
    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=False)
    for frame in frames:
        writer.write(frame)
    writer.release()
    convert_to_all_intra_h264(raw_path)
    os.replace(raw_path, output_path)
    if save_preview_png:
        save_preview(first_frame, output_path.with_suffix(".png"))


def write_color_video(frames: list[np.ndarray], output_path: Path, fps: int, save_preview_png: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    convert_to_all_intra_h264(raw_path)
    os.replace(raw_path, output_path)
    if save_preview_png:
        save_preview(first_frame, output_path.with_suffix(".png"))


def render_l1_demo(mode: str, output_path: Path, sample_id: str, save_preview_png: bool = True) -> dict:
    sample = load_sample_by_id(L1_METADATA, sample_id)
    generator = L1TextCaptchaGenerator()
    text_mask = generator.build_text_mask(sample["text"])
    tex_fg = generate_dot_texture(
        L1_VIDEO_HEIGHT,
        L1_VIDEO_WIDTH,
        L1_DOT_COVERAGE,
        L1_DOT_RADIUS,
        stable_seed(sample["id"], mode, "l1_fg"),
    )
    tex_bg = generate_dot_texture(
        L1_VIDEO_HEIGHT,
        L1_VIDEO_WIDTH,
        L1_DOT_COVERAGE,
        L1_DOT_RADIUS,
        stable_seed(sample["id"], mode, "l1_bg"),
    )

    frames: list[np.ndarray] = []
    for frame_idx in range(L1_TOTAL_FRAMES):
        curr_fg = np.roll(tex_fg, int(frame_idx * L1_FG_SPEED), axis=1)
        if mode == "foreground_only":
            frame = np.zeros((L1_VIDEO_HEIGHT, L1_VIDEO_WIDTH), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported ablation mode: {mode}")
        frame[text_mask] = curr_fg[text_mask]
        frames.append(frame)

    write_gray_video(frames, output_path, L1_FPS, save_preview_png=save_preview_png)
    return {
        "level": "L1",
        "sample_id": sample["id"],
        "target": sample["text"],
        "source_video": sample["video_path"],
        "ablation_video": str(output_path.relative_to(ROOT_DIR)),
        "preview_png": str(output_path.with_suffix(".png").relative_to(ROOT_DIR)) if save_preview_png else None,
        "notes": {
            "foreground_speed": L1_FG_SPEED,
            "background_speed": None,
            "background_removed": mode == "foreground_only",
            "text_length": L1_TEXT_LENGTH,
        },
    }


def render_l2_demo(mode: str, output_path: Path, sample_id: str, save_preview_png: bool = True) -> dict:
    sample = load_sample_by_id(L2_METADATA, sample_id)
    obj_path = Path(sample["target"]["obj_path"])
    scene_tm = load_normalized_scene(obj_path)
    mesh = scene_to_single_mesh(scene_tm)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    camera_pose = build_three_quarter_camera_pose(L2_CAMERA_DISTANCE)
    world_to_camera = np.linalg.inv(camera_pose).astype(np.float32)
    yfov = np.pi / 3.0
    focal = 0.5 * L2_VIDEO_HEIGHT / np.tan(yfov / 2.0)
    cx = L2_VIDEO_WIDTH / 2.0
    cy = L2_VIDEO_HEIGHT / 2.0

    def rotation_y(angle: float, scale: float = 1.0) -> np.ndarray:
        c = np.cos(angle) * scale
        s = np.sin(angle) * scale
        return np.array(
            [
                [c, 0.0, s],
                [0.0, scale, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float32,
        )

    def project_vertices(rotated_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        homog = np.concatenate([rotated_vertices, np.ones((rotated_vertices.shape[0], 1), dtype=np.float32)], axis=1)
        cam = homog @ world_to_camera.T
        depth = -cam[:, 2]
        proj = np.zeros((rotated_vertices.shape[0], 2), dtype=np.float32)
        valid = depth > 1e-4
        proj[valid, 0] = focal * (cam[valid, 0] / depth[valid]) + cx
        proj[valid, 1] = -focal * (cam[valid, 1] / depth[valid]) + cy
        return proj, valid

    def projected_extent(scale: float, angle: float) -> float:
        rotated = vertices @ rotation_y(angle, scale=scale).T
        proj, valid = project_vertices(rotated)
        if not np.any(valid):
            return 0.0
        pts = proj[valid]
        return float(max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()))

    sampled_extent = 0.0
    for sample_idx in range(12):
        sampled_extent = max(sampled_extent, projected_extent(1.0, 2.0 * np.pi * sample_idx / 12.0))
    target_extent = float(min(L2_VIDEO_WIDTH, L2_VIDEO_HEIGHT) * 0.78)
    mesh_scale = target_extent / max(sampled_extent, 1.0)
    mesh_scale = float(np.clip(mesh_scale, 0.55, 2.2))

    def render_mask(angle: float) -> np.ndarray:
        rotated = vertices @ rotation_y(angle, scale=mesh_scale).T
        proj, valid_vertices = project_vertices(rotated)
        mask = np.zeros((L2_VIDEO_HEIGHT, L2_VIDEO_WIDTH), dtype=np.uint8)
        for tri in faces:
            if not (valid_vertices[tri[0]] and valid_vertices[tri[1]] and valid_vertices[tri[2]]):
                continue
            pts = np.round(proj[tri]).astype(np.int32)
            cv2.fillConvexPoly(mask, pts, 255)
        return mask

    tex_fg = generate_dot_texture(
        L2_VIDEO_HEIGHT,
        L2_VIDEO_WIDTH,
        L2_FG_COVERAGE,
        L2_DOT_RADIUS,
        stable_seed(sample["id"], mode, "l2_fg"),
    )

    frames: list[np.ndarray] = []
    degrees_per_frame = 360.0 / L2_TOTAL_FRAMES
    for frame_idx in range(L2_TOTAL_FRAMES):
        angle = np.radians(frame_idx * degrees_per_frame)
        mask = render_mask(angle) > 0

        curr_fg = np.roll(tex_fg, frame_idx * L2_FG_SPEED, axis=1)
        if mode == "foreground_only":
            frame = np.zeros((L2_VIDEO_HEIGHT, L2_VIDEO_WIDTH), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported ablation mode: {mode}")
        frame[mask] = curr_fg[mask]
        frames.append(frame)

    write_gray_video(frames, output_path, L2_FPS, save_preview_png=save_preview_png)
    return {
        "level": "L2",
        "sample_id": sample["id"],
        "target": sample["target"]["category"],
        "source_video": sample["video_path"],
        "ablation_video": str(output_path.relative_to(ROOT_DIR)),
        "preview_png": str(output_path.with_suffix(".png").relative_to(ROOT_DIR)) if save_preview_png else None,
        "notes": {
            "foreground_speed": L2_FG_SPEED,
            "background_speed": None,
            "background_removed": mode == "foreground_only",
            "obj_path": str(obj_path),
            "render_proxy": "software_projected_mesh_silhouette",
        },
    }


def render_l3_demo(mode: str, output_path: Path, sample_id: str, save_preview_png: bool = True) -> dict:
    import smplx
    import torch
    from MVCAP_code.L3.generate_l3_from_selection_rigid_drift import (
        BACKGROUND_POINT_COUNT,
        CHALLENGE_DRAW_SCALE,
        CHALLENGE_HEIGHT,
        CHALLENGE_OFFSET_Y,
        CHALLENGE_WIDTH,
        FPS as L3_FPS,
        JOINT_INDICES_17,
        MODEL_ROOT,
        load_segment,
    )

    sample = load_sample_by_id(L3_METADATA, sample_id)
    npz_path = ROOT_DIR / "dataset" / "L3_raw" / sample["local_relpath"]
    poses, trans, betas = load_segment(npz_path, sample["start_t"], sample["end_t"])
    frame_count = int(poses.shape[0])
    model = smplx.create(
        str(MODEL_ROOT),
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        use_pca=False,
        batch_size=frame_count,
    )
    with torch.no_grad():
        output = model(
            global_orient=poses[:, :3],
            body_pose=poses[:, 3:],
            betas=betas.expand(frame_count, -1),
            transl=trans,
        )
    joints = output.joints[:, JOINT_INDICES_17, :]
    pts_2d = joints[..., [0, 2]]

    rng = np.random.default_rng(stable_seed(sample["id"], mode, "l3_bg"))
    background_points = rng.integers(
        low=np.array([0, 0]),
        high=np.array([CHALLENGE_WIDTH, CHALLENGE_HEIGHT]),
        size=(BACKGROUND_POINT_COUNT, 2),
    ).astype(np.int32)
    drift_points = background_points.copy()

    frames: list[np.ndarray] = []
    for frame_idx in range(frame_count):
        img = np.zeros((CHALLENGE_HEIGHT, CHALLENGE_WIDTH, 3), dtype=np.uint8)
        if mode == "foreground_only":
            pass
        else:
            raise ValueError(f"Unsupported ablation mode: {mode}")

        for pt in pts_2d[frame_idx].cpu().numpy():
            px = int(pt[0] * CHALLENGE_DRAW_SCALE + CHALLENGE_WIDTH // 2)
            py = int(-pt[1] * CHALLENGE_DRAW_SCALE + CHALLENGE_OFFSET_Y)
            cv2.circle(img, (px, py), 3, (255, 255, 255), -1)

        frames.append(img)

    write_color_video(frames, output_path, L3_FPS, save_preview_png=save_preview_png)
    return {
        "level": "L3",
        "sample_id": sample["id"],
        "target": sample["category"],
        "source_video": f"dataset/L3_selected_generated_rigid_drift/library/{sample['challenge_video']}",
        "ablation_video": str(output_path.relative_to(ROOT_DIR)),
        "preview_png": str(output_path.with_suffix(".png").relative_to(ROOT_DIR)) if save_preview_png else None,
        "notes": {
            "background_point_count": 0,
            "background_removed": mode == "foreground_only",
            "local_relpath": sample["local_relpath"],
        },
    }


def load_selected_example_ids(config_path: Path) -> dict[str, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = ["L1", "L2", "L3"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing example ids in {config_path}: {missing}")
    return {key: str(config[key]) for key in required}


def official_200_metadata_paths() -> dict[str, Path]:
    return {
        "L1": L1_200_METADATA,
        "L2": L2_200_METADATA,
        "L3": L3_200_METADATA,
    }


def generate_demo_bundle(mode: str, base_dir: Path, example_ids: dict[str, str]) -> Path:
    if mode not in ABLATION_SPECS:
        raise ValueError(f"Unknown ablation mode: {mode}")

    base_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "ablation_mode": mode,
        "ablation_code": ABLATION_SPECS[mode]["code"],
        "title": ABLATION_SPECS[mode]["title"],
        "description": ABLATION_SPECS[mode]["description"],
        "selected_example_ids": example_ids,
        "levels": {},
    }

    l1_out = outputs_dir / f"{ABLATION_SPECS[mode]['code']}_L1_demo.mp4"
    l2_out = outputs_dir / f"{ABLATION_SPECS[mode]['code']}_L2_demo.mp4"
    l3_out = outputs_dir / f"{ABLATION_SPECS[mode]['code']}_L3_demo.mp4"

    manifest["levels"]["L1"] = render_l1_demo(mode, l1_out, example_ids["L1"])
    manifest["levels"]["L2"] = render_l2_demo(mode, l2_out, example_ids["L2"])
    manifest["levels"]["L3"] = render_l3_demo(mode, l3_out, example_ids["L3"])

    manifest_path = base_dir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def build_l1_ablation_record(sample: dict, output_path: Path, metadata_root: Path, mode: str) -> dict:
    record = deepcopy(sample)
    record["video_path"] = str(output_path.relative_to(ROOT_DIR))
    record["background_mode"] = mode
    record["ablation_mode"] = ABLATION_SPECS[mode]["code"]
    record["source_video_path"] = sample["video_path"]
    record["source_metadata"] = str(L1_200_METADATA.relative_to(ROOT_DIR))
    return record


def build_l2_ablation_record(sample: dict, output_path: Path, metadata_root: Path, mode: str) -> dict:
    record = deepcopy(sample)
    record["video_path"] = str(output_path.relative_to(ROOT_DIR))
    record["background_mode"] = mode
    record["ablation_mode"] = ABLATION_SPECS[mode]["code"]
    record["source_video_path"] = sample["video_path"]
    record["source_metadata"] = str(L2_200_METADATA.relative_to(ROOT_DIR))
    return record


def build_l3_ablation_record(sample: dict, output_path: Path, metadata_root: Path, mode: str) -> dict:
    record = deepcopy(sample)
    record["challenge_video"] = output_path.name
    record["background_mode"] = mode
    record["ablation_mode"] = ABLATION_SPECS[mode]["code"]
    record["source_challenge_video"] = sample["challenge_video"]
    record["source_metadata"] = str(L3_200_METADATA.relative_to(ROOT_DIR))
    return record


def generate_foreground_only_dataset(
    base_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    levels: tuple[str, ...] = ("L1", "L2", "L3"),
) -> dict[str, Path]:
    mode = "foreground_only"
    spec = ABLATION_SPECS[mode]
    dataset_root = base_dir / "datasets"
    l1_root = dataset_root / "L1_text_dataset_foreground_only"
    l2_root = dataset_root / "L2_shapenet_foreground_only"
    l3_root = dataset_root / "L3_selected_generated_foreground_only"
    l1_root.mkdir(parents=True, exist_ok=True)
    (l2_root / "videos").mkdir(parents=True, exist_ok=True)
    (l3_root / "library").mkdir(parents=True, exist_ok=True)

    l1_samples = load_samples(L1_200_METADATA) if "L1" in levels else []
    l2_samples = load_samples(L2_200_METADATA) if "L2" in levels else []
    l3_samples = load_samples(L3_200_METADATA) if "L3" in levels else []
    if limit is not None:
        l1_samples = l1_samples[:limit]
        l2_samples = l2_samples[:limit]
        l3_samples = l3_samples[:limit]

    l1_records: list[dict] = []
    for idx, sample in enumerate(l1_samples, start=1):
        output_path = l1_root / f"{sample['id']}.mp4"
        if overwrite or not output_path.exists():
            render_l1_demo(mode, output_path, sample["id"], save_preview_png=False)
        l1_records.append(build_l1_ablation_record(sample, output_path, l1_root, mode))
        print(f"[A][L1] {idx}/{len(l1_samples)} {sample['id']}", flush=True)

    l2_records: list[dict] = []
    for idx, sample in enumerate(l2_samples, start=1):
        output_path = l2_root / "videos" / f"{sample['id']}.mp4"
        if overwrite or not output_path.exists():
            render_l2_demo(mode, output_path, sample["id"], save_preview_png=False)
        l2_records.append(build_l2_ablation_record(sample, output_path, l2_root, mode))
        print(f"[A][L2] {idx}/{len(l2_samples)} {sample['id']}", flush=True)

    l3_records: list[dict] = []
    for idx, sample in enumerate(l3_samples, start=1):
        output_path = l3_root / "library" / sample["challenge_video"]
        if overwrite or not output_path.exists():
            render_l3_demo(mode, output_path, sample["id"], save_preview_png=False)
        l3_records.append(build_l3_ablation_record(sample, output_path, l3_root, mode))
        print(f"[A][L3] {idx}/{len(l3_samples)} {sample['id']}", flush=True)

    l1_meta = l1_root / "L1_200_metadata.json"
    l2_meta = l2_root / "L2_200_metadata.json"
    l3_meta = l3_root / "L3_200_metadata.json"
    records = {"L1": l1_records, "L2": l2_records, "L3": l3_records}
    metadata = {"L1": l1_meta, "L2": l2_meta, "L3": l3_meta}
    for level in levels:
        metadata[level].write_text(json.dumps(records[level], indent=2), encoding="utf-8")

    manifest = {
        "ablation_mode": mode,
        "ablation_code": spec["code"],
        "description": spec["description"],
        "source_metadata": {
            k: str(v.relative_to(ROOT_DIR))
            for k, v in official_200_metadata_paths().items() if k in levels
        },
        "generated_metadata": {level: str(metadata[level].relative_to(ROOT_DIR)) for level in levels},
        "counts": {level: len(records[level]) for level in levels},
    }
    manifest_path = dataset_root / "foreground_only_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, **{level: metadata[level] for level in levels}}
