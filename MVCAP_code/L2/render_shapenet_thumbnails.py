import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrender
from PIL import Image

from .config import SELECTION_JSON, THUMB_HEIGHT, THUMB_ROOT, THUMB_WIDTH
from .shapenet_utils import build_thumbnail_scene, load_normalized_scene, load_selection


TARGET_OCCUPANCY = 0.78


def normalize_rgba_canvas(rgba: np.ndarray, target_occupancy: float = TARGET_OCCUPANCY) -> np.ndarray:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rgba

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cropped = rgba[y0:y1, x0:x1]

    h, w = cropped.shape[:2]
    if h == 0 or w == 0:
        return rgba

    target_size = int(min(THUMB_WIDTH, THUMB_HEIGHT) * target_occupancy)
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((THUMB_HEIGHT, THUMB_WIDTH, 4), dtype=np.uint8)

    x_start = (THUMB_WIDTH - new_w) // 2
    y_start = (THUMB_HEIGHT - new_h) // 2
    canvas[y_start:y_start + new_h, x_start:x_start + new_w] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ShapeNetCore thumbnails for L2 options.")
    parser.add_argument("--selection-path", type=str, default=str(SELECTION_JSON))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selection = load_selection(Path(args.selection_path))
    renderer = pyrender.OffscreenRenderer(THUMB_WIDTH, THUMB_HEIGHT)

    total = sum(len(items) for items in selection.values())
    rendered = 0

    for category, items in selection.items():
        out_dir = THUMB_ROOT / category
        out_dir.mkdir(parents=True, exist_ok=True)
        for item_idx, item in enumerate(items):
            sample_name = item.get("sample_name", f"{category}_{item_idx:02d}")
            png_path = out_dir / f"{sample_name}.png"
            if png_path.exists() and not args.overwrite:
                continue
            scene_tm = load_normalized_scene(Path(item["obj_path"]))
            scene = build_thumbnail_scene(scene_tm)
            color, _ = renderer.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
            rgba = normalize_rgba_canvas(np.asarray(color))
            Image.fromarray(rgba, "RGBA").save(png_path)
            rendered += 1
            if rendered % 10 == 0 or rendered == total:
                print(f"Rendered {rendered}/{total} thumbnails")

    renderer.delete()
    print(f"Thumbnails available at {THUMB_ROOT}")


if __name__ == "__main__":
    main()
