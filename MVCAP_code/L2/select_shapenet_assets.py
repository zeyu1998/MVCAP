import argparse
import json
import random

from .config import DEFAULT_NUM_PER_CATEGORY, SEED, SELECTION_JSON, SELECTION_SUMMARY_JSON
from .shapenet_utils import category_records, ensure_l2_dirs, extract_asset, iter_zip_assets, zip_path_for_synset


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and extract ShapeNetCore assets for L2 generation.")
    parser.add_argument("--num-per-category", type=int, default=DEFAULT_NUM_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true", help="Re-extract already selected assets.")
    args = parser.parse_args()

    ensure_l2_dirs()
    rng = random.Random(args.seed)

    selection = {}
    summary = {}

    for category, info in category_records():
        zip_path = zip_path_for_synset(info["synset"])
        assets = iter_zip_assets(zip_path)
        if not assets:
            raise RuntimeError(f"No valid assets found in {zip_path}")

        ranked = assets[:]
        top_pool = ranked[: max(args.num_per_category * 3, args.num_per_category)]
        rng.shuffle(top_pool)
        chosen = sorted(top_pool[: args.num_per_category], key=lambda item: item["model_id"])

        extracted = []
        for item_idx, item in enumerate(chosen):
            extracted_item = extract_asset(item, overwrite=args.overwrite)
            extracted_item["sample_index"] = item_idx
            extracted_item["sample_name"] = f"{category}_{item_idx:02d}"
            extracted.append(extracted_item)
        selection[category] = extracted
        summary[category] = {
            "synset": info["synset"],
            "label": info["label"],
            "available_assets": len(assets),
            "selected_assets": len(extracted),
            "with_textures": sum(1 for item in extracted if item["texture_count"] > 0),
        }
        print(
            f"{category}: selected {len(extracted)} / {len(assets)} "
            f"(textures in selection: {summary[category]['with_textures']})"
        )

    SELECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_JSON.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    SELECTION_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved selection to {SELECTION_JSON}")
    print(f"Saved summary to {SELECTION_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
