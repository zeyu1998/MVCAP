# L3: biological motion

L3 creates a sparse 17-point challenge and a mesh-rendered action option.
The main challenge has dynamic background distractors; FG removes them.

## Inputs and environment

```bash
python -m pip install -r requirements-l3.txt
# Needed when loading legacy SMPL pickle files that contain chumpy objects:
python -m pip install --no-build-isolation chumpy==0.70
export MVCAP_SMPL_ROOT=/absolute/path/to/body_models
```

The model root should contain `smpl/SMPL_NEUTRAL.pkl` as expected by `smplx`.
If unset, the default root is `assets/body_models/`, so the default full path is
`assets/body_models/smpl/SMPL_NEUTRAL.pkl`. Download and prepare the model through its
official provider. The Python package does not include model parameters.

Prepare the separately obtained local inputs:

```text
dataset/babel_v1.0_release/train.json, val.json, test.json
dataset/L3_raw/CMU/<subject>/<motion>_poses.npz
```

The loader expects AMASS-style fields: `poses`, `betas`, `trans`, and
`mocap_frame_rate`. Original CMU ASF/AMC files cannot be passed directly to
this loader. See [THIRD_PARTY.md](../../THIRD_PARTY.md) for source and license
information. Rendering also requires a working OpenGL context and FFmpeg.

## Select and generate

Run from the repository root:

```bash
python -m MVCAP_code.L3.select_babel_l3_actions --per-class 20 --allow-shortfall
python -m MVCAP_code.L3.generate_l3_from_selection_rigid_drift --seed 42
```

The selector targets 13 actions, excluding `sit`, favors long/high-frame
segments, and avoids sharing source files between classes where possible.
`--allow-shortfall` retains fewer examples when coverage is insufficient.
The historical filename `dataset/L3_babel/l3_14class_selection.json` is kept
for compatibility; it does not imply 14 current action classes.

The generator clips the motion, downsamples toward 30 FPS, interpolates short
sequences, and center-crops longer sequences to 90 frames. It removes global
x/z translation, renders challenge and option videos, and assembles a target
plus five distractors from different classes. At least six populated action
classes are required for quiz assembly.

Outputs are written to `dataset/L3_selected_generated_rigid_drift/library/`
and `L3_metadata.json`. The initially selected pool is not guaranteed to
contain exactly 200 samples. Freeze a curated subset explicitly if needed.

To rebuild quizzes from a curated local metadata list:

```bash
python -m MVCAP_code.L3.assemble_l3_curated_quiz_metadata \
  --input-path dataset/L3_selected_generated_rigid_drift/L3_metadata.json \
  --output-path dataset/L3_selected_generated_rigid_drift/L3_200_metadata.json
```

This command assembles questions; it does not itself select exactly 200 rows.
It keeps `kick` and `exercise/training` out of each other's distractor sets.

## Other regimes and FG

```bash
python -m MVCAP_code.L3.generate_l3_from_selection_perturbed_drift --seed 42
python -m MVCAP_code.L3.generate_l3_from_selection_random_walk --seed 42
python -m ablation_study.A.generate_full_dataset --levels L3
python -m ablation_study.A.generate_vqa_files --levels L3
```

The alternative generators reuse the local selection and write separate
regime folders. FG reuses source motion segments, semantic choices, and the
main option videos. The generated videos use all-intra H.264 for frame access.
