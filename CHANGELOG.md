# Changelog

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
