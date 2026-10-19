# L1: coherent motion

L1 reveals four-character text through relative motion between foreground
and background dot fields. It requires only `requirements.txt` and FFmpeg.

Run from the repository root:

```bash
python -m MVCAP_code.L1.generate_l1_dataset_rigid_drift --num-samples 200 --seed 42
```

Outputs: challenge MP4s, foreground answer PNGs, and
`dataset/L1_text_dataset_rigid_drift/dataset_metadata.json`.
Each metadata record contains `target_option` and five `distractors`.

Alternative background regimes:

```bash
python -m MVCAP_code.L1.generate_l1_dataset_perturbed_drift --num-samples 200 --seed 42
python -m MVCAP_code.L1.generate_l1_dataset_random_walk --num-samples 200 --seed 42
```

They write to `dataset/L1_text_dataset_perturbed_drift/` and
`dataset/L1_text_dataset_random_walk/` respectively. Use `--help` for speed
overrides and `--overwrite` for a complete regeneration.

The standard geometry and dot settings are in `config.py`. Font availability
affects the mask. For cross-machine reproduction, use the same font and
software versions.

Generate the matched control after the rigid-drift run:

```bash
python -m ablation_study.A.generate_full_dataset --levels L1
python -m ablation_study.A.generate_vqa_files --levels L1
```
