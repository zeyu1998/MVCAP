# MVCAP-Bench-FG generation

The matched foreground-only control removes the dynamic background camouflage
and retains the source sample ID, semantic target, and distractor metadata.
It rerenders the foreground; it does not extract pixels from the main MP4.

Run from the repository root after generating the main condition:

```bash
python -m ablation_study.A.generate_full_dataset --levels L1 L2 L3
python -m ablation_study.A.generate_vqa_files --levels L1 L2 L3
```

Use `--levels L1` for an asset-free run, `--limit 3` for a small preview,
and `--overwrite` to regenerate existing FG videos.

## Source metadata

| Level | Preferred frozen subset | Fallback from the main generator |
|---|---|---|
| L1 | `dataset/L1_text_dataset_rigid_drift/L1_200_metadata.json` | `dataset_metadata.json` in that directory |
| L2 | `dataset/L2_shapenet_rigid_drift/L2_200_metadata.json` | `L2_metadata.json` in that directory |
| L3 | `dataset/L3_selected_generated_rigid_drift/L3_200_metadata.json` | `L3_metadata.json` in that directory |

FG videos and metadata are written under `ablation_study/A/datasets/`, with
level-specific `L*_200_metadata.json` files and `foreground_only_manifest.json`.
The `_200` suffix is historical and is retained even for smaller smoke runs.
The manifest records the actual counts.

L2 requires the original local meshes referenced by the source metadata.
L3 requires the original motion files under `dataset/L3_raw/` and the
separately obtained SMPL model. L1 does not import the L3 rendering stack.

## Choices and labels

Source metadata is copied before the challenge path and condition fields
are updated. Ground-truth semantics, candidate identities, and the source
candidate order remain available. Option PNGs/videos remain in the main
dataset and are reused rather than duplicated.

The annotation exporter shuffles a fresh six-choice list with seed `42 +
sample_index` and writes `L*_200_vqa.json` with A–F labels. Therefore a main
and FG export may have different answer letters. Score each condition using
its own `correct_option_label`, and keep answer fields out of model inputs.
