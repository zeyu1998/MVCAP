# Generation parameters

Defaults in this source release:

| Setting | L1 | L2 | L3 |
|---|---|---|---|
| Semantic task | 4-character text | 13 object classes | 13 action classes |
| Main challenge size | 488 × 488 | 448 × 448 | 448 × 448 |
| Frame rate | 30 FPS | 30 FPS | 30 FPS |
| Duration / frames | 3 s / 90 | 3 s / 90 | 3 s / 90 |
| Foreground representation | Text dot mask | Rotating mesh silhouette in dots | 17 body joints |
| Foreground / background carrier | 0.15 dot coverage each | 0.15 dot coverage each | 17 foreground joints / 170 background points |
| Rigid background displacement | +2 px/frame along x | +2 px/frame along x | −4 px/frame along x |
| Foreground texture displacement | −2 px/frame along x | −2 px/frame along x | Derived from articulated motion |
| Dot radius | 2 px | 2 px | 3 px |
| Choices | 1 target + 5 distractors | 1 target + 5 distractors | 1 target + 5 distractors |
| Seed | 42 | 42 | 42 |
| Option representation | Text / answer PNG | 512 × 512 thumbnail | 256 × 256 mesh video |

L1 excludes easily confused characters from the alphabet:
`ABCDEFGHJKLMNPQRSTUVWXYZ23456789`. L2 normalizes meshes, rotates one full
turn over the challenge, and uses a 0.78 target occupancy ratio. L3 option
rendering uses a global video bounding box with a target height ratio of
0.65 and width cap of 0.8.

The default generation count is 200 for L1/L2. L3 starts with up to 20
segments per class and allows source-coverage shortfalls; curated benchmark
subsets are selected separately.

## Entry points

- `MVCAP_code/L1/config.py`: L1 constants.
- `MVCAP_code/L2/config.py`: L2 constants and category/synset map.
- `MVCAP_code/L3/select_babel_l3_actions.py`: action list, duration bounds,
  frame-count ranking, and source-file selection.
- `MVCAP_code/L3/generate_l3_from_selection_rigid_drift.py`: L3 dimensions,
  joint list, background count, motion loading, and option rendering.
- `ablation_study/common.py`: FG rendering and stable per-sample seeds.

Use each module's `--help` to inspect overrides. L1/L2 perturbation and
random-walk dynamics are defined in their corresponding generator files;
they are alternative regimes, not parameters of the rigid-drift benchmark.

## Pairing boundaries

FG keeps the semantic sample and candidates while removing background dots.
FG texture seeds are derived from sample IDs and the condition. In L2, the
FG mask uses the existing software-projection path rather than the main
OpenGL path. Exact pixels are therefore not asserted to be identical between
main and FG; pairing refers to semantic targets and experimental controls.
