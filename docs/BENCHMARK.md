# MVCAP-Bench: public offline benchmark

**[Download MVCAP-Bench v1.0 (ZIP, 1.82 GB)](https://github.com/zeyu1998/MVCAP/releases/download/bench-v1.0/mvcap-bench-offline-v1.0.zip)**

[Release page](https://github.com/zeyu1998/MVCAP/releases/tag/bench-v1.0) ·
[Archive SHA-256](https://github.com/zeyu1998/MVCAP/releases/download/bench-v1.0/mvcap-bench-offline-v1.0.zip.sha256) ·
[Archive validation](https://github.com/zeyu1998/MVCAP/releases/download/bench-v1.0/mvcap-bench-offline-v1.0.archive-validation.json)

The download is public and does not require an email request. This is the
existing fixed offline data package, with original media copied byte-for-byte,
prepared on 2026-10-13 and publicly released on 2026-10-21.

## Contents

| Level | Task | Paired tasks | Main videos | FG videos | Option media |
|---|---|---:|---:|---:|---|
| L1 | Coherent motion / four-character text | 200 | 200 | 200 | Text |
| L2 | Structural motion / object category | 200 | 200 | 200 | Text and 260 shared thumbnails |
| L3 | Biological motion / human action | 200 | 200 | 200 | Text and 200 shared action videos |
| Total | | 600 | 600 | 600 | Six choices per condition |

MVCAP-Bench is the rigid-drift main condition; MVCAP-Bench-FG is its matched
foreground-only control. These are 600 paired semantic tasks across two
conditions. The archive contains 1,660 media files, including 1,200 challenge
videos and the shared option assets.

```text
mvcap-bench-offline-v1.0/
  annotations/L1.json, L2.json, L3.json
  media/main/L1/, L2/, L3/
  media/fg/L1/, L2/, L3/
  media/options/L2/, L3/
  pairs.csv
  manifest.json
  media_manifest.json
  provenance.json
  validation_report.json
  SHA256SUMS
  CITATION.bib
  README.md
```

## Download and verify

```bash
curl -L --fail -o mvcap-bench-offline-v1.0.zip \
  https://github.com/zeyu1998/MVCAP/releases/download/bench-v1.0/mvcap-bench-offline-v1.0.zip
curl -L --fail -o mvcap-bench-offline-v1.0.zip.sha256 \
  https://github.com/zeyu1998/MVCAP/releases/download/bench-v1.0/mvcap-bench-offline-v1.0.zip.sha256
shasum -a 256 -c mvcap-bench-offline-v1.0.zip.sha256
unzip mvcap-bench-offline-v1.0.zip
```

On Linux, `sha256sum -c` can replace `shasum -a 256 -c`.
The archive is 1,817,401,432 bytes. Its SHA-256 is:

```text
64993f768b37403c62913495278949cf2cc2d483cc5cc16263625eb00ecdfe4d
```

## Read a task

Paths in the annotations are relative to the extracted package root.

```python
import json
from pathlib import Path

root = Path("mvcap-bench-offline-v1.0")
sample = json.loads((root / "annotations/L2.json").read_text())[0]
condition = "main"  # or "fg"
task = sample[condition]
video = root / task["challenge_video"]
question = sample["question"]
options = task["options"]
expected = sample["ground_truth"][f"{condition}_correct_option_label"]
```

Each condition has six A-F choices and its own correct answer letter. Main
and FG may have different candidate orders, so score using the answer letter
for the chosen condition. Provide the challenge, question, and that condition's
options to a model. Keep ground truth, `pairs.csv`, and `provenance.json` on
the evaluation side.

The archive retains the original offline VQA prompts, candidate text, option
order, and ground truth. L2/L3 also provide option media for visual matching.
Report the option modality, frame sampling, resizing, and inference settings
used. Offline VQA accuracy and browser-interaction results are separate settings.

## Integrity, sources, and citation

Preparation checks cover all 600 pairs, condition-specific answers, media
references, byte equality with source files, and full decoding of the included
media. The package contains detailed validation and per-media checksums.
After extraction, the package's `SHA256SUMS` verifies all other package files.

L2 uses ShapeNet-derived stimuli. L3 uses BABEL/CMU/AMASS-derived motion and
SMPL-based body rendering. Source IDs are retained for traceability. Raw meshes,
motion arrays, downloaded BABEL files, SMPL parameters, credentials, agent logs,
and participant records are not included. See [THIRD_PARTY.md](../THIRD_PARTY.md)
for source acknowledgments and the applicable upstream terms. The MIT code
license does not replace those terms.

Please cite **Invisible in Space, Visible in Time: Motion Vision CAPTCHA against
GUI Agents** (ACM Multimedia 2026), [CITATION.bib](../CITATION.bib), and the relevant
source resources.
