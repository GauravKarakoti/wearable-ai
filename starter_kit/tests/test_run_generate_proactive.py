#!/usr/bin/env python3
"""Unit tests for run_generate_proactive.py (ECCV 2026 Starter Kit).

Mocks `model.create_model` and `model.extract_frames` so the inference flow
can be exercised on CPU without GPU/model/video files.

Tests cover:
  - _normalize_dialog_turns: role normalization + empty-text handling
  - Cumulative frame extraction: chunk j gets frames from intervals[0..j]
    (capped by --max-frames)
  - Dialog history slicing under --max-history-turns (incl. 0 and -1)
  - Message construction: system + query (when present) + history
  - Output JSONL structure: one {video_path, answers} per session
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

STARTER_KIT_DIR = os.environ.get(
    "STARTER_KIT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.insert(0, STARTER_KIT_DIR)

import model as _model_module  # noqa: E402
import run_generate_proactive as rg  # noqa: E402


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class _MockModel:
    """Records every generate() call; returns a deterministic prediction."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __enter__(self) -> "_MockModel":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        return False

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        self.calls.append(
            {
                "n_frames": len(frames),
                "max_new_tokens": max_new_tokens,
                "messages": [
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
            }
        )
        idx = len(self.calls) - 1
        return f"$interrupt$ chunk{idx}" if idx % 2 == 0 else "$silent$"


def _install_mocks(mock_model: _MockModel) -> None:
    """Patch the `model` module so the lazy import inside main() picks up
    the mocked factory, frame extractor, and GPU setup."""
    _model_module.create_model = lambda *_args, **_kwargs: mock_model
    _model_module.setup_gpus = lambda **_kwargs: 0

    def _fake_extract_frames(
        video_path: str,
        intervals: list[tuple[float, float]] | None = None,
        frames_per_interval: int = 4,
        max_frames: int = 32,
    ) -> list[object]:
        n = frames_per_interval * len(intervals or [])
        return [None] * min(n, max_frames)

    _model_module.extract_frames = _fake_extract_frames


def _write_jsonl(path: str, rows: list[dict]) -> str:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _make_session(
    n_chunks: int = 4,
    query: str = "Help me make espresso step by step",
    task: str = "Cooking",
    extra_dialog_at_chunk: dict[int, list[dict]] | None = None,
) -> dict:
    """Synthetic session JSONL row.

    `extra_dialog_at_chunk[i]` is appended after the initial query turn,
    simulating prior decisions/turns recorded before chunk i.
    """
    extra = extra_dialog_at_chunk or {}
    return {
        "video_path": "fake.mp4",
        "video_intervals": [[i * 8.0, (i + 1) * 8.0] for i in range(n_chunks)],
        "query": query,
        "task": task,
        "dialog": [
            [{"role": "user", "text": query}, *extra.get(i, [])]
            for i in range(n_chunks)
        ],
        "answers": ["$silent$"] * n_chunks,  # placeholder; not used at inference
    }


# ---------------------------------------------------------------------------
# _normalize_dialog_turns (pure function)
# ---------------------------------------------------------------------------


class NormalizeDialogTurnsTest(unittest.TestCase):
    def test_user_and_assistant_roles_preserved(self) -> None:
        result = rg._normalize_dialog_turns(
            [
                {"role": "user", "text": "hi"},
                {"role": "assistant", "text": "hello"},
            ]
        )
        self.assertEqual(
            result,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )

    def test_unknown_role_defaults_to_user(self) -> None:
        result = rg._normalize_dialog_turns([{"role": "system", "text": "x"}])
        self.assertEqual(result, [{"role": "user", "content": "x"}])

    def test_role_case_insensitive(self) -> None:
        result = rg._normalize_dialog_turns(
            [
                {"role": "ASSISTANT", "text": "hi"},
                {"role": "User", "text": "yo"},
            ]
        )
        self.assertEqual(result[0]["role"], "assistant")
        self.assertEqual(result[1]["role"], "user")

    def test_empty_text_is_skipped(self) -> None:
        result = rg._normalize_dialog_turns(
            [
                {"role": "user", "text": ""},
                {"role": "assistant", "text": "kept"},
            ]
        )
        self.assertEqual(result, [{"role": "assistant", "content": "kept"}])

    def test_empty_input(self) -> None:
        self.assertEqual(rg._normalize_dialog_turns([]), [])


# ---------------------------------------------------------------------------
# End-to-end (mocked) base class
# ---------------------------------------------------------------------------


class _MockedRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(
        self,
        args_list: list[str],
        sessions: list[dict],
    ) -> tuple[list[dict], _MockModel]:
        """Invoke `run_generate_proactive.main()` with mocked model.

        Returns (predictions, mock_model). Predictions is a list of
        {video_path, answers} parsed from the output JSONL.
        """
        mock = _MockModel()
        _install_mocks(mock)
        in_path = _write_jsonl(os.path.join(self.tmpdir, "in.jsonl"), sessions)
        out_path = os.path.join(self.tmpdir, "out.jsonl")
        full_argv = [
            "run_generate_proactive.py",
            "--input",
            in_path,
            "--output",
            out_path,
            "--video-folder",
            self.tmpdir,
            "--no-eval",
            *args_list,
        ]
        with patch.object(sys, "argv", full_argv):
            rg.main()
        with open(out_path) as f:
            preds = [json.loads(line) for line in f if line.strip()]
        return preds, mock


# ---------------------------------------------------------------------------
# Cumulative frame extraction
# ---------------------------------------------------------------------------


class CumulativeFramesTest(_MockedRunTest):
    def test_frames_grow_with_chunk_index(self) -> None:
        _, mock = self._run(
            [
                "--frames-per-interval",
                "16",
                "--max-frames",
                "256",
                "--max-history-turns",
                "-1",
            ],
            [_make_session(n_chunks=4)],
        )
        self.assertEqual(len(mock.calls), 4)
        self.assertEqual(mock.calls[0]["n_frames"], 16)
        self.assertEqual(mock.calls[1]["n_frames"], 32)
        self.assertEqual(mock.calls[2]["n_frames"], 48)
        self.assertEqual(mock.calls[3]["n_frames"], 64)

    def test_max_frames_caps_total(self) -> None:
        _, mock = self._run(
            [
                "--frames-per-interval",
                "16",
                "--max-frames",
                "32",
                "--max-history-turns",
                "-1",
            ],
            [_make_session(n_chunks=4)],
        )
        # Once the cumulative count exceeds 32, the script subsamples down.
        self.assertEqual(mock.calls[0]["n_frames"], 16)
        self.assertEqual(mock.calls[1]["n_frames"], 32)
        self.assertEqual(mock.calls[2]["n_frames"], 32)
        self.assertEqual(mock.calls[3]["n_frames"], 32)


# ---------------------------------------------------------------------------
# History slicing under --max-history-turns
# ---------------------------------------------------------------------------


class HistorySlicingTest(_MockedRunTest):
    def _session_with_history(self) -> dict:
        # dialog[3] = [query, A, U, A] -> 3 past turns after the query.
        return _make_session(
            n_chunks=4,
            extra_dialog_at_chunk={
                1: [{"role": "assistant", "text": "$interrupt$ Grind"}],
                2: [{"role": "assistant", "text": "$interrupt$ Grind"}],
                3: [
                    {"role": "assistant", "text": "$interrupt$ Grind"},
                    {"role": "user", "text": "Coarse or fine?"},
                    {"role": "assistant", "text": "$interrupt$ Fine"},
                ],
            },
        )

    def _history_messages(self, mock: _MockModel, chunk: int) -> list[dict]:
        # Strip system + query (first 2 messages) to isolate the history.
        return mock.calls[chunk]["messages"][2:]

    def test_minus_one_keeps_all_history(self) -> None:
        _, mock = self._run(
            ["--max-history-turns", "-1"], [self._session_with_history()]
        )
        history3 = self._history_messages(mock, 3)
        self.assertEqual(len(history3), 3)
        self.assertEqual(history3[-1]["content"], "$interrupt$ Fine")

    def test_zero_keeps_no_past_turns(self) -> None:
        _, mock = self._run(
            ["--max-history-turns", "0"], [self._session_with_history()]
        )
        for chunk in range(4):
            self.assertEqual(self._history_messages(mock, chunk), [])

    def test_one_keeps_only_most_recent(self) -> None:
        _, mock = self._run(
            ["--max-history-turns", "1"], [self._session_with_history()]
        )
        history3 = self._history_messages(mock, 3)
        self.assertEqual(len(history3), 1)
        self.assertEqual(history3[0]["content"], "$interrupt$ Fine")

    def test_two_keeps_last_two(self) -> None:
        _, mock = self._run(
            ["--max-history-turns", "2"], [self._session_with_history()]
        )
        history3 = self._history_messages(mock, 3)
        self.assertEqual(
            [m["content"] for m in history3],
            ["Coarse or fine?", "$interrupt$ Fine"],
        )

    def test_large_value_keeps_all_available(self) -> None:
        _, mock = self._run(
            ["--max-history-turns", "99"], [self._session_with_history()]
        )
        # Only 3 past turns exist at chunk 3; slicing should not error.
        self.assertEqual(len(self._history_messages(mock, 3)), 3)


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


class MessageConstructionTest(_MockedRunTest):
    def test_first_message_is_system(self) -> None:
        _, mock = self._run(["--max-history-turns", "-1"], [_make_session()])
        self.assertEqual(mock.calls[0]["messages"][0]["role"], "system")
        # System prompt should mention the duplex tokens.
        self.assertIn("$interrupt$", mock.calls[0]["messages"][0]["content"])
        self.assertIn("$silent$", mock.calls[0]["messages"][0]["content"])

    def test_query_added_as_first_user_turn(self) -> None:
        session = _make_session(query="Help me make espresso")
        _, mock = self._run(["--max-history-turns", "-1"], [session])
        self.assertEqual(mock.calls[0]["messages"][1]["role"], "user")
        self.assertEqual(
            mock.calls[0]["messages"][1]["content"], "Help me make espresso"
        )

    def test_empty_query_means_no_user_turn_appended(self) -> None:
        # _make_session uses query as the first user turn in dialog[i] too;
        # an empty query yields an empty-text turn that _normalize drops.
        session = _make_session(query="")
        _, mock = self._run(["--max-history-turns", "-1"], [session])
        self.assertEqual(len(mock.calls[0]["messages"]), 1)
        self.assertEqual(mock.calls[0]["messages"][0]["role"], "system")


# ---------------------------------------------------------------------------
# Output JSONL structure
# ---------------------------------------------------------------------------


class OutputJsonlTest(_MockedRunTest):
    def test_one_pred_row_per_session(self) -> None:
        sessions = [
            _make_session(n_chunks=2, query="q1"),
            _make_session(n_chunks=3, query="q2"),
        ]
        # Differentiate video_path so we can assert ordering.
        sessions[0]["video_path"] = "v1.mp4"
        sessions[1]["video_path"] = "v2.mp4"
        preds, _ = self._run(["--max-history-turns", "0"], sessions)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0]["video_path"], "v1.mp4")
        self.assertEqual(preds[1]["video_path"], "v2.mp4")
        self.assertEqual(len(preds[0]["answers"]), 2)
        self.assertEqual(len(preds[1]["answers"]), 3)

    def test_pred_answers_are_non_empty_strings(self) -> None:
        preds, _ = self._run(["--max-history-turns", "0"], [_make_session(n_chunks=3)])
        for ans in preds[0]["answers"]:
            self.assertIsInstance(ans, str)
            self.assertGreater(len(ans), 0)


if __name__ == "__main__":
    unittest.main()
