# Release validation

Validation performed for the generation-code release on 2026-10-19.

- Four asset-free tests passed with `python -m unittest discover -s tests -v`.
  These run real L1 main/FG encoding and decoding, compare seeded generation
  across processes with different Python hash seeds, and check L3 quiz
  choices against the semantic exclusion rules.
- All 16 command-line entry points passed `--help`.
- L2 was exercised with 13 locally generated synthetic box meshes: archive
  selection/extraction, all option thumbnails, one challenge in each motion
  regime, FG generation, and six-choice annotation export.
- L3 was exercised with six synthetic motion sequences and a separately held
  neutral SMPL model: main challenge and option rendering, alternative
  challenge regimes with `--skip-options`, FG generation, and annotation export.
- All 29 MP4s produced by the L2/L3 smoke checks decoded completely with
  FFmpeg. L2/L3 target/distractor identities were preserved in FG metadata.

The rendering checks used Python 3.12.13, NumPy 1.26.4, OpenCV 4.10.0,
Pillow 11.3.0, pyrender 0.1.45, pyglet 2.1.19, trimesh 4.12.2, smplx 0.1.28,
and a locally installed PyTorch on macOS/Apple Silicon. Legacy SMPL pickle
loading used chumpy 0.70. The older pyglet 1.5 series encountered a repeated
context-creation failure in this environment; the rendering requirements
therefore pin the tested pyglet version.

Synthetic inputs exercise the generation and annotation pipeline. They do
not validate recognition accuracy, action semantics, the licensed source
selection, or exact reproduction of the frozen publication benchmark.
External input files, model parameters, and smoke-test outputs are not
distributed in this source repository.
