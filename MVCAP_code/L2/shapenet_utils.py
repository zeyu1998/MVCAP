import json
import zipfile
from pathlib import Path

import numpy as np

from .config import (
    ASSET_ROOT,
    CAMERA_AZIMUTH_DEG,
    CAMERA_ELEVATION_DEG,
    SHAPENET_ROOT,
    TARGET_CATEGORIES,
    THUMB_CAMERA_DISTANCE,
)


def ensure_l2_dirs() -> None:
    for path in [ASSET_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def zip_path_for_synset(synset: str) -> Path:
    return SHAPENET_ROOT / f"{synset}.zip"


def iter_zip_assets(zip_path: Path) -> list[dict]:
    assets = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if len(parts) < 3:
                continue
            synset = parts[0]
            model_id = parts[1]
            rel = "/".join(parts[2:])
            entry = assets.setdefault(
                model_id,
                {
                    "synset": synset,
                    "model_id": model_id,
                    "has_obj": False,
                    "has_mtl": False,
                    "texture_count": 0,
                    "image_count": 0,
                    "member_prefix": f"{synset}/{model_id}/",
                },
            )
            if rel == "models/model_normalized.obj":
                entry["has_obj"] = True
            elif rel == "models/model_normalized.mtl":
                entry["has_mtl"] = True
            elif rel.startswith("images/") and rel.lower().endswith((".jpg", ".jpeg", ".png")):
                entry["image_count"] += 1
                entry["texture_count"] += 1
            elif rel.startswith("models/") and rel.lower().endswith((".jpg", ".jpeg", ".png")):
                entry["texture_count"] += 1

    valid = [item for item in assets.values() if item["has_obj"]]
    valid.sort(key=lambda item: (-item["texture_count"], -item["image_count"], item["model_id"]))
    return valid


def extract_asset(asset: dict, overwrite: bool = False) -> dict:
    synset = asset["synset"]
    model_id = asset["model_id"]
    zip_path = zip_path_for_synset(synset)
    target_dir = ASSET_ROOT / synset / model_id
    obj_path = target_dir / "models" / "model_normalized.obj"

    if overwrite or not obj_path.exists():
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            prefix = asset["member_prefix"]
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel = Path(member).relative_to(Path(prefix))
                if str(rel) == ".":
                    continue
                out_path = target_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if member.endswith("/"):
                    out_path.mkdir(parents=True, exist_ok=True)
                    continue
                with zf.open(member) as src, out_path.open("wb") as dst:
                    dst.write(src.read())

    extracted = dict(asset)
    extracted["asset_dir"] = str(target_dir)
    extracted["obj_path"] = str(obj_path)
    return extracted


def category_records() -> list[tuple[str, dict]]:
    return list(TARGET_CATEGORIES.items())


def load_normalized_scene(obj_path: Path):
    import trimesh

    scene_or_mesh = trimesh.load(obj_path, force="scene")
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh.copy()
        scene_tm = trimesh.Scene(mesh)
    else:
        scene_tm = scene_or_mesh.copy()

    if scene_tm.is_empty:
        raise ValueError(f"Empty scene: {obj_path}")

    scale = 1.0 / float(np.max(scene_tm.extents))
    center_tf = np.eye(4)
    center_tf[:3, 3] = -scene_tm.centroid
    scene_tm.apply_transform(center_tf)
    scene_tm.apply_scale(scale)
    return scene_tm


def scene_to_single_mesh(scene_tm):
    import trimesh

    try:
        mesh = scene_tm.dump(concatenate=True)
    except Exception:
        geom = list(scene_tm.geometry.values())
        if not geom:
            raise ValueError("No geometry available in scene")
        mesh = trimesh.util.concatenate(geom)
    return mesh


def build_three_quarter_camera_pose(distance: float) -> np.ndarray:
    azimuth = np.radians(CAMERA_AZIMUTH_DEG)
    elevation = np.radians(CAMERA_ELEVATION_DEG)

    eye = np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.cos(azimuth),
        ],
        dtype=np.float32,
    )
    eye = eye / np.linalg.norm(eye) * distance

    target = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    camera_pose = np.eye(4)
    camera_pose[:3, 0] = x_axis
    camera_pose[:3, 1] = y_axis
    camera_pose[:3, 2] = z_axis
    camera_pose[:3, 3] = eye
    return camera_pose


def build_thumbnail_scene(scene_tm):
    import pyrender

    scene = pyrender.Scene.from_trimesh_scene(scene_tm, bg_color=[0, 0, 0, 0])
    scene.ambient_light = [0.85, 0.85, 0.85]

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = build_three_quarter_camera_pose(THUMB_CAMERA_DISTANCE)

    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)
    return scene


def load_selection(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
