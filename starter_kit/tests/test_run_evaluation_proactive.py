#!/usr/bin/env python3
"""Unit tests for the Proactive eval path in run_evaluation.py (ECCV 2026 Starter Kit).

Tests cover:
  - parse_tag: $interrupt$ / $silent$ token classification
  - binary_metrics: per-class precision/recall/F1 + macro F1 + g-mean F1
  - score_proactive: full scoring over a list of sessions, with per-task breakdown
  - load_jsonl: JSONL file loading
  - _resolve_path: relative path resolution
  - End-to-end: full pipeline through `run_evaluation.py --task proactive --eval-only`
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

STARTER_KIT_DIR = os.environ.get(
    "STARTER_KIT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
EVAL_SCRIPT = os.path.join(STARTER_KIT_DIR, "run_evaluation.py")

sys.path.insert(0, STARTER_KIT_DIR)

import run_evaluation as ev  # noqa: E402


def _write_jsonl(path: str, rows: list[dict]) -> str:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _session(video_path: str, task: str, gold: list[str]) -> dict:
    return {
        "video_path": video_path,
        "task": task,
        "video_intervals": [[i * 8.0, (i + 1) * 8.0] for i in range(len(gold))],
        "answers": gold,
    }


class ParseTagTest(unittest.TestCase):
    def test_interrupt_with_utterance(self) -> None:
        self.assertEqual(ev.parse_tag("$interrupt$ Add salt now."), "interrupt")

    def test_interrupt_no_space(self) -> None:
        self.assertEqual(ev.parse_tag("$interrupt$Add salt now."), "interrupt")

    def test_silent(self) -> None:
        self.assertEqual(ev.parse_tag("$silent$"), "silent")

    def test_leading_whitespace_interrupt(self) -> None:
        self.assertEqual(ev.parse_tag("   $interrupt$ hello"), "interrupt")

    def test_leading_whitespace_silent(self) -> None:
        self.assertEqual(ev.parse_tag("\n\t$silent$"), "silent")

    def test_empty_string_is_silent(self) -> None:
        self.assertEqual(ev.parse_tag(""), "silent")

    def test_arbitrary_text_is_silent(self) -> None:
        # Anything not starting with $interrupt$ collapses to silent.
        self.assertEqual(ev.parse_tag("Sure, here is the answer"), "silent")

    def test_interrupt_anywhere_else_is_silent(self) -> None:
        # Token must be at the (lstripped) start.
        self.assertEqual(ev.parse_tag("Sure, $interrupt$ hi"), "silent")


class BinaryMetricsTest(unittest.TestCase):
    def test_perfect_all_interrupt(self) -> None:
        m = ev.binary_metrics(tp=10, fp=0, tn=0, fn=0)
        self.assertEqual(m["interrupt_f1"], 1.0)
        # No silent samples -> silent_f1 is 0 by convention.
        self.assertEqual(m["silent_f1"], 0.0)
        self.assertEqual(m["macro_f1"], 0.5)
        # G-mean of (1.0, 0.0) is 0 — sharper signal of class asymmetry.
        self.assertEqual(m["gmean_f1"], 0.0)

    def test_perfect_balanced(self) -> None:
        m = ev.binary_metrics(tp=5, fp=0, tn=5, fn=0)
        self.assertEqual(m["interrupt_f1"], 1.0)
        self.assertEqual(m["silent_f1"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["gmean_f1"], 1.0)

    def test_all_wrong(self) -> None:
        m = ev.binary_metrics(tp=0, fp=5, tn=0, fn=5)
        self.assertEqual(m["interrupt_f1"], 0.0)
        self.assertEqual(m["silent_f1"], 0.0)
        self.assertEqual(m["macro_f1"], 0.0)
        self.assertEqual(m["gmean_f1"], 0.0)

    def test_known_mixed_values(self) -> None:
        # tp=7, fp=3, tn=5, fn=3 -> matches synthetic-data hand calcs.
        m = ev.binary_metrics(tp=7, fp=3, tn=5, fn=3)
        self.assertEqual(m["interrupt_precision"], 0.7)
        self.assertEqual(m["interrupt_recall"], 0.7)
        self.assertEqual(m["interrupt_f1"], 0.7)
        self.assertEqual(m["silent_precision"], 0.625)
        self.assertEqual(m["silent_recall"], 0.625)
        self.assertEqual(m["silent_f1"], 0.625)
        self.assertEqual(m["macro_f1"], round((0.7 + 0.625) / 2, 4))
        # G-mean F1 = sqrt(0.7 * 0.625) = sqrt(0.4375) ~= 0.6614
        self.assertEqual(m["gmean_f1"], round(math.sqrt(0.7 * 0.625), 4))
        self.assertEqual(m["support"], 18)

    def test_gmean_punishes_class_asymmetry(self) -> None:
        # Asymmetric: int_F1=0.9, sil_F1=0.1 -> macro_f1=0.5, gmean_f1=0.3
        # vs. balanced int_F1=sil_F1=0.5 -> macro_f1=0.5, gmean_f1=0.5
        # G-mean penalises the asymmetric case despite identical macro F1.
        # tp=9, fn=1, fp=1, tn=1 (skewed): int_p=int_r=0.9 so int_F1=0.9.
        # sil_p=tn/(tn+fn)=1/2=0.5, sil_r=tn/(tn+fp)=1/2=0.5, sil_F1=0.5.
        m = ev.binary_metrics(tp=9, fp=1, tn=1, fn=1)
        self.assertEqual(m["interrupt_f1"], 0.9)
        self.assertEqual(m["silent_f1"], 0.5)
        self.assertEqual(m["macro_f1"], 0.7)
        self.assertEqual(m["gmean_f1"], round(math.sqrt(0.9 * 0.5), 4))
        # gmean (~0.6708) is below macro (0.7) when classes differ.
        self.assertLess(m["gmean_f1"], m["macro_f1"])

    def test_empty(self) -> None:
        m = ev.binary_metrics(tp=0, fp=0, tn=0, fn=0)
        self.assertEqual(m["macro_f1"], 0.0)
        self.assertEqual(m["gmean_f1"], 0.0)
        self.assertEqual(m["support"], 0)


class ScoreProactiveTest(unittest.TestCase):
    def test_perfect_match(self) -> None:
        golden = [_session("a.mp4", "Cooking", ["$interrupt$ x", "$silent$"])]
        preds = [{"video_path": "a.mp4", "answers": ["$interrupt$ y", "$silent$"]}]
        results = ev.score_proactive(golden, preds)
        self.assertEqual(results["overall"]["macro_f1"], 1.0)
        self.assertEqual(results["total_sessions"], 1)
        self.assertEqual(results["skipped_chunks"], 0)
        self.assertIn("Cooking", results["per_task"])
        self.assertEqual(results["per_task"]["Cooking"]["macro_f1"], 1.0)

    def test_known_mixed_metrics(self) -> None:
        # 5 sessions, 18 chunks, 2 tasks; matches end-to-end synthetic data.
        # tp=7, fp=3, tn=5, fn=3 overall.
        cooking_gold = ["$interrupt$ a", "$silent$", "$interrupt$ b", "$silent$"]
        cooking_pred = ["$interrupt$ a", "$silent$", "$silent$", "$interrupt$ x"]
        sight_gold = ["$interrupt$ a", "$interrupt$ b", "$silent$"]
        sight_pred = ["$interrupt$ a", "$interrupt$ b", "$silent$"]

        golden = [
            _session(f"cook{i}.mp4", "Cooking", cooking_gold) for i in range(3)
        ] + [_session(f"craft{i}.mp4", "Crafts", sight_gold) for i in range(2)]
        preds = [
            {"video_path": f"cook{i}.mp4", "answers": cooking_pred} for i in range(3)
        ] + [{"video_path": f"craft{i}.mp4", "answers": sight_pred} for i in range(2)]

        results = ev.score_proactive(golden, preds)
        self.assertEqual(results["overall"]["macro_f1"], round((0.7 + 0.625) / 2, 4))
        self.assertEqual(results["overall"]["support"], 18)

        self.assertEqual(results["per_task"]["Cooking"]["support"], 12)
        self.assertEqual(results["per_task"]["Cooking"]["macro_f1"], 0.5)
        self.assertEqual(results["per_task"]["Crafts"]["support"], 6)
        self.assertEqual(results["per_task"]["Crafts"]["macro_f1"], 1.0)

    def test_length_mismatch_skips_extra_chunks(self) -> None:
        golden = [_session("a.mp4", "X", ["$interrupt$ x", "$silent$", "$silent$"])]
        preds = [{"video_path": "a.mp4", "answers": ["$interrupt$ x", "$silent$"]}]
        results = ev.score_proactive(golden, preds)
        # Only the first 2 chunks are scored.
        self.assertEqual(results["overall"]["support"], 2)
        self.assertEqual(results["overall"]["macro_f1"], 1.0)
        self.assertEqual(results["skipped_chunks"], 1)

    def test_missing_task_defaults_to_unknown(self) -> None:
        golden = [{"video_path": "a.mp4", "answers": ["$interrupt$ x"]}]
        preds = [{"video_path": "a.mp4", "answers": ["$interrupt$ y"]}]
        results = ev.score_proactive(golden, preds)
        self.assertIn("unknown", results["per_task"])

    def test_session_count_mismatch_raises(self) -> None:
        golden = [_session("a.mp4", "X", ["$silent$"])]
        preds = []
        with self.assertRaises(ValueError):
            ev.score_proactive(golden, preds)

    def test_all_silent_baseline(self) -> None:
        # Pred is always silent; gold has a mix.
        golden = [_session("a.mp4", "X", ["$interrupt$ x", "$silent$", "$silent$"])]
        preds = [
            {"video_path": "a.mp4", "answers": ["$silent$", "$silent$", "$silent$"]}
        ]
        results = ev.score_proactive(golden, preds)
        # The 1 interrupt is missed -> recall=0 on interrupt; gmean collapses.
        self.assertEqual(results["overall"]["interrupt_recall"], 0.0)
        self.assertEqual(results["overall"]["interrupt_f1"], 0.0)
        self.assertEqual(results["overall"]["gmean_f1"], 0.0)


class LoadJsonlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_basic(self) -> None:
        path = os.path.join(self.tmpdir, "x.jsonl")
        _write_jsonl(path, [{"a": 1}, {"a": 2}])
        rows = ev.load_jsonl(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"a": 1})

    def test_skips_blank_lines(self) -> None:
        path = os.path.join(self.tmpdir, "x.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"a": 1}) + "\n\n")
            f.write(json.dumps({"a": 2}) + "\n")
        rows = ev.load_jsonl(path)
        self.assertEqual(len(rows), 2)


class ResolvePathTest(unittest.TestCase):
    def test_absolute_passthrough(self) -> None:
        self.assertEqual(ev._resolve_path("/abs/path"), "/abs/path")

    def test_relative_is_resolved(self) -> None:
        result = ev._resolve_path("data/foo.jsonl")
        self.assertTrue(os.path.isabs(result))
        self.assertTrue(
            result.endswith(os.path.join("starter_kit", "data", "foo.jsonl"))
        )


class EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_main_writes_results(self) -> None:
        golden_path = _write_jsonl(
            os.path.join(self.tmpdir, "g.jsonl"),
            [_session("a.mp4", "Cooking", ["$interrupt$ x", "$silent$"])],
        )
        preds_path = _write_jsonl(
            os.path.join(self.tmpdir, "p.jsonl"),
            [{"video_path": "a.mp4", "answers": ["$interrupt$ y", "$silent$"]}],
        )
        out_path = os.path.join(self.tmpdir, "results.json")

        result = subprocess.run(
            [
                sys.executable,
                EVAL_SCRIPT,
                "--task",
                "proactive",
                "--eval-only",
                "--golden",
                golden_path,
                "--predictions",
                preds_path,
                "--eval-output",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        with open(out_path) as f:
            results = json.load(f)
        self.assertEqual(results["overall"]["macro_f1"], 1.0)
        self.assertEqual(results["total_sessions"], 1)
        self.assertIn("Cooking", results["per_task"])

    def test_main_errors_on_session_count_mismatch(self) -> None:
        golden_path = _write_jsonl(
            os.path.join(self.tmpdir, "g.jsonl"),
            [
                _session("a.mp4", "X", ["$silent$"]),
                _session("b.mp4", "X", ["$silent$"]),
            ],
        )
        preds_path = _write_jsonl(
            os.path.join(self.tmpdir, "p.jsonl"),
            [{"video_path": "a.mp4", "answers": ["$silent$"]}],
        )
        out_path = os.path.join(self.tmpdir, "results.json")

        result = subprocess.run(
            [
                sys.executable,
                EVAL_SCRIPT,
                "--task",
                "proactive",
                "--eval-only",
                "--golden",
                golden_path,
                "--predictions",
                preds_path,
                "--eval-output",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("ERROR", result.stderr + result.stdout)

    def test_main_handles_empty_inputs_gracefully(self) -> None:
        # Both empty -> 0 sessions, 0 chunks, success exit.
        golden_path = _write_jsonl(os.path.join(self.tmpdir, "g.jsonl"), [])
        preds_path = _write_jsonl(os.path.join(self.tmpdir, "p.jsonl"), [])
        out_path = os.path.join(self.tmpdir, "results.json")

        result = subprocess.run(
            [
                sys.executable,
                EVAL_SCRIPT,
                "--task",
                "proactive",
                "--eval-only",
                "--golden",
                golden_path,
                "--predictions",
                preds_path,
                "--eval-output",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        with open(out_path) as f:
            results = json.load(f)
        self.assertEqual(results["total_sessions"], 0)
        self.assertEqual(results["overall"]["support"], 0)


if __name__ == "__main__":
    unittest.main()
