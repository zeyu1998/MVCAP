"""Exercise generated videos and pairing without external datasets or models."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2

from MVCAP_code.L3.assemble_l3_curated_quiz_metadata import build_quiz_metadata


ROOT = Path(__file__).resolve().parents[1]


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mvcap-generation-test-")
        self.root = Path(self.temporary.name)
        for folder in ("MVCAP_code", "ablation_study"):
            shutil.copytree(ROOT / folder, self.root / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "datasets"))
        self.addCleanup(self.temporary.cleanup)

    def run_module(self, module, *args, hash_seed="123"):
        result = subprocess.run(
            [sys.executable, "-m", module, *args], cwd=self.root,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True, capture_output=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def decoded_video(self, path):
        capture = cv2.VideoCapture(str(path))
        self.assertTrue(capture.isOpened(), str(path))
        self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30, places=2)
        digest = hashlib.sha256()
        count = 0
        total = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            self.assertEqual(frame.shape[:2], (488, 488))
            digest.update(frame.tobytes())
            total += int(frame.sum())
            count += 1
        capture.release()
        self.assertEqual(count, 90)
        self.assertGreater(total, 0)
        return digest.hexdigest(), total

    def test_l1_seed_is_stable_across_processes(self):
        module = "MVCAP_code.L1.generate_l1_dataset_rigid_drift"
        self.run_module(module, "--num-samples", "2", "--seed", "42")
        root = self.root / "dataset/L1_text_dataset_rigid_drift"
        first = json.loads((root / "dataset_metadata.json").read_text())
        first_digest, _ = self.decoded_video(root / "L1_000.mp4")
        for sample in first:
            candidates = [sample["target_option"], *sample["distractors"]]
            self.assertEqual(len({item["text"] for item in candidates}), 6)
            self.assertEqual(sample["target_option"]["text"], sample["text"])

        self.run_module(module, "--num-samples", "2", "--seed", "42", "--overwrite",
                        hash_seed="987")
        repeated = json.loads((root / "dataset_metadata.json").read_text())
        self.assertEqual(first, repeated)
        self.assertEqual(first_digest, self.decoded_video(root / "L1_000.mp4")[0])

    def test_fg_preserves_choices_and_decodes_with_less_background(self):
        self.run_module("MVCAP_code.L1.generate_l1_dataset_rigid_drift",
                        "--num-samples", "1", "--seed", "42")
        main_root = self.root / "dataset/L1_text_dataset_rigid_drift"
        source = json.loads((main_root / "dataset_metadata.json").read_text())[0]
        self.run_module("ablation_study.A.generate_full_dataset", "--levels", "L1", "--limit", "1")
        self.run_module("ablation_study.A.generate_vqa_files", "--levels", "L1")
        fg_root = self.root / "ablation_study/A/datasets/L1_text_dataset_foreground_only"
        fg = json.loads((fg_root / "L1_200_metadata.json").read_text())[0]
        for field in ("id", "text", "target_option", "distractors"):
            self.assertEqual(source[field], fg[field])
        self.assertEqual(fg["background_mode"], "foreground_only")
        _, main_sum = self.decoded_video(self.root / source["video_path"])
        _, fg_sum = self.decoded_video(self.root / fg["video_path"])
        self.assertLess(fg_sum, main_sum)
        vqa = json.loads((fg_root / "L1_200_vqa.json").read_text())[0]
        self.assertEqual(list(vqa["option_map"]), list("ABCDEF"))
        self.assertEqual(vqa["option_map"][vqa["correct_option_label"]], source["text"])

    def test_alternative_l1_regimes_generate_complete_videos(self):
        for regime in ("perturbed_drift", "random_walk"):
            with self.subTest(regime=regime):
                self.run_module(f"MVCAP_code.L1.generate_l1_dataset_{regime}",
                                "--num-samples", "1", "--seed", "42")
                root = self.root / f"dataset/L1_text_dataset_{regime}"
                sample = json.loads((root / "dataset_metadata.json").read_text())[0]
                self.assertEqual(sample["background_mode"], regime)
                self.decoded_video(self.root / sample["video_path"])


class QuizSemanticsTest(unittest.TestCase):
    def test_l3_choices_preserve_targets_and_avoid_conflicting_actions(self):
        categories = ["kick", "exercise/training", "walk", "run", "jump", "dance", "wave"]
        pool = [
            {"id": f"sample_{i}", "category": category, "challenge_video": f"c_{i}.mp4",
             "option_video": f"o_{i}.mp4", "local_relpath": None, "start_t": 0,
             "end_t": 3, "background_mode": "rigid_drift"}
            for i, category in enumerate(categories)
        ]
        untouched = copy.deepcopy(pool)
        quiz = build_quiz_metadata(pool, 5)
        self.assertEqual(pool, untouched)
        self.assertEqual(quiz, build_quiz_metadata(pool, 5))
        for sample in quiz:
            choices = [sample["target_option"], *sample["distractors"]]
            self.assertEqual(len({item["category"] for item in choices}), 6)
            self.assertEqual(sample["target_option"]["id"], sample["id"])
            self.assertFalse({"kick", "exercise/training"}.issubset(
                {item["category"] for item in choices}))


if __name__ == "__main__":
    unittest.main()
