# Changelog

## MVCAP-Bench v1.0 - 2026-10-21

- Public GitHub Release download of the original paired offline benchmark:
  600 tasks across L1/L2/L3, 1,200 main/FG challenge videos, option media,
  six-choice annotations, ground truth, and source mappings.
- Archive checksum, validation report, and a dataset loading guide.
- Direct download links replace the email-only dataset access instructions.

## Unreleased

### Added

- Three looping examples with Semantic Target in the first column, followed
  by synchronized MVCAP-Bench and MVCAP-Bench-FG panels. The L2 object rotates
  and the L3 mesh body performs the matching motion in the first column.
- Standalone Semantic Target GIFs and 30 fps videos for L2/L3.
- A reproducible custom `CAPTCHA` text illustration, plus motorbike and kicking
  examples from the frozen benchmark.
- Full-size 30 fps video companions, sample provenance, and source acknowledgments.
- An email contact for requesting the complete offline benchmark.

## 1.0.0 — 2026-10-19

### Added

- L1 coherent motion, L2 structural motion, and L3 biological motion generators.
- Rigid-drift main generation and the existing perturbed-drift/random-walk variants.
- Matched foreground-only generation, six-choice annotations, and level selection.
- Setup instructions, parameter documentation, data/model acquisition notes,
  and an asset-free L1 pipeline smoke test.

### Changed for standalone use

- Seed NumPy alongside the Python RNG; sort L1 distractor strings for stable
  metadata across Python hash seeds.
- Read ShapeNet and SMPL input locations from optional environment variables.
- Allow FG generation to use freshly generated metadata when no frozen
  `*_200_metadata.json` subset is supplied.
- Import L3 dependencies only when L3 foreground-only generation is requested.

This release preserves the generation methods, but rerunning it is not a
claim of byte-identical reproduction of previously rendered benchmark media.
