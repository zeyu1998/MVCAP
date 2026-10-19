import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SHAPENET_ROOT = Path(os.environ.get("MVCAP_SHAPENET_ROOT", ROOT_DIR / "mv_source" / "ShapeNetCore")).expanduser()

DATASET_ROOT = ROOT_DIR / "dataset" / "L2_shapenet_rigid_drift"
ASSET_ROOT = DATASET_ROOT / "assets"
THUMB_ROOT = DATASET_ROOT / "thumbnails"
VIDEO_ROOT = DATASET_ROOT / "videos"
SELECTION_JSON = DATASET_ROOT / "shapenet_selection.json"
SELECTION_SUMMARY_JSON = DATASET_ROOT / "shapenet_selection_summary.json"
METADATA_JSON = DATASET_ROOT / "L2_metadata.json"

SEED = 42
DEFAULT_NUM_PER_CATEGORY = 20

TARGET_CATEGORIES = {
    "AIRPLANE": {"synset": "02691156", "label": "airplane", "color": "#4E79A7"},
    "BED": {"synset": "02818832", "label": "bed", "color": "#EDC948"},
    "TRASH_BIN": {"synset": "02747177", "label": "trash bin", "color": "#9C755F"},
    "CAR": {"synset": "02958343", "label": "car", "color": "#E15759"},
    "CHAIR": {"synset": "03001627", "label": "chair", "color": "#FF9DA7"},
    "GUITAR": {"synset": "03467517", "label": "guitar", "color": "#B07AA1"},
    "LAMP": {"synset": "03636649", "label": "lamp", "color": "#BAB0AC"},
    "LAPTOP": {"synset": "03642806", "label": "laptop", "color": "#76B7B2"},
    "MOTORBIKE": {"synset": "03790512", "label": "motorbike", "color": "#AF7AA1"},
    "PLANTER": {"synset": "03991062", "label": "flowerpot", "color": "#86BCB6"},
    "SOFA": {"synset": "04256520", "label": "sofa", "color": "#E17C05"},
    "TABLE": {"synset": "04379243", "label": "table", "color": "#79706E"},
    "WATERCRAFT": {"synset": "04530566", "label": "watercraft", "color": "#F28E2B"},
}

DEFAULT_NUM_SAMPLES = 200

THUMB_WIDTH = 512
THUMB_HEIGHT = 512

VIDEO_WIDTH = 448
VIDEO_HEIGHT = 448
FPS = 30
DURATION = 3.0
TOTAL_FRAMES = int(FPS * DURATION)
VIDEO_TARGET_OCCUPANCY = 0.78

DOT_RADIUS = 2
FG_COVERAGE = 0.15
BG_COVERAGE = 0.15
BG_SPEED = 2
FG_SPEED = -2

CAMERA_DISTANCE = 1.3
THUMB_CAMERA_DISTANCE = 1.8
CAMERA_AZIMUTH_DEG = 45.0
CAMERA_ELEVATION_DEG = 24.0
