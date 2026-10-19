# L2: structural motion

L2 reveals a rotating 3D object in a dot field. The included ShapeNet pipeline
selects assets, renders static option thumbnails, and generates six-way
challenges. It targets 13 object categories listed in `config.py`.

## Inputs and environment

```bash
python -m pip install -r requirements-rendering.txt
export MVCAP_SHAPENET_ROOT=/absolute/path/to/ShapeNetCore
```

Obtain the source from the official provider under the applicable terms;
see [THIRD_PARTY.md](../../THIRD_PARTY.md). The archive directory should contain
one `<synset>.zip` per category, with members such as
`02691156/<model_id>/models/model_normalized.obj` and associated resources.
The default input directory is `mv_source/ShapeNetCore/` if the environment
variable is unset. A working OpenGL rendering backend and FFmpeg are required.

## Main pipeline

Run these commands from the repository root, in order:

```bash
python -m MVCAP_code.L2.select_shapenet_assets --num-per-category 20 --seed 42
python -m MVCAP_code.L2.render_shapenet_thumbnails
python -m MVCAP_code.L2.generate_shapenet_l2_dataset_rigid_drift --num-samples 200 --seed 42
```

The selection prefers assets with textures, randomizes a top-ranked candidate
pool, and records the selected model IDs. Outputs are under
`dataset/L2_shapenet_rigid_drift/`: `assets/`, `thumbnails/`, `videos/`,
`shapenet_selection.json`, and `L2_metadata.json`.

Reuse the selection for other regimes:

```bash
python -m MVCAP_code.L2.generate_shapenet_l2_dataset_perturbed_drift --num-samples 200 --seed 42
python -m MVCAP_code.L2.generate_shapenet_l2_dataset_random_walk --num-samples 200 --seed 42
```

They write to separate `L2_shapenet_perturbed_drift` and `L2_shapenet_random_walk`
directories. Use `--selection-path` to supply an existing local selection.

## Foreground-only control

```bash
python -m ablation_study.A.generate_full_dataset --levels L2
python -m ablation_study.A.generate_vqa_files --levels L2
```

FG uses the same selected mesh and semantic choices. Its existing rendering
path uses a software-projected mesh silhouette, with no background dots.
Thumbnails remain in the main dataset. This is a regenerated control, not a
pixel-subtraction operation on an already encoded main video.
