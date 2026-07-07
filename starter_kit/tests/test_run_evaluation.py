#!/usr/bin/env python3
"""Unit tests for run_evaluation.py (ECCV 2026 Starter Kit).

Tests cover:
  - normalize_answer: MCQ answer extraction from various formats
  - _sentence_bleu: Self-contained BLEU scoring with add-1 smoothing
  - _ngrams: N-gram counting helper
  - _parse_judge_score: LLM judge output parsing
  - _build_judge_prompt: Judge prompt construction
  - load_jsonl: JSONL file loading
  - evaluate_longqa: Full LongQA MCQ evaluation
  - evaluate_convqa: Full ConvQA evaluation (BLEU only, no LLM judge in unit tests)
  - compute_bleu_scores: Per-turn BLEU computation
  - CLI argument parsing: --task, --golden, --predictions, --llm-judge, etc.
  - Error handling: mismatched lengths, empty inputs, missing fields
  - E2E: full pipeline from JSONL files through evaluation to JSON output
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest.mock

import pytest


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

STARTER_KIT_DIR = os.environ.get(
    "STARTER_KIT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
EVAL_SCRIPT = os.path.join(STARTER_KIT_DIR, "run_evaluation.py")

sys.path.insert(0, STARTER_KIT_DIR)
import run_evaluation as ev


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def longqa_golden_data():
    return [
        {
            "video_path": "v1.mp4",
            "question": "What color?",
            "mcq_answer": "A",
            "category": "color",
        },
        {
            "video_path": "v2.mp4",
            "question": "How many?",
            "mcq_answer": "B",
            "category": "count",
        },
        {
            "video_path": "v3.mp4",
            "question": "Where?",
            "mcq_answer": "C",
            "category": "color",
        },
        {
            "video_path": "v4.mp4",
            "question": "When?",
            "mcq_answer": "D",
            "category": "time",
        },
    ]


@pytest.fixture
def longqa_preds_perfect(longqa_golden_data):
    return [{"mcq_answer": g["mcq_answer"]} for g in longqa_golden_data]


@pytest.fixture
def longqa_preds_half(longqa_golden_data):
    preds = []
    for i, g in enumerate(longqa_golden_data):
        if i % 2 == 0:
            preds.append({"mcq_answer": g["mcq_answer"]})
        else:
            wrong = "A" if g["mcq_answer"] != "A" else "B"
            preds.append({"mcq_answer": wrong})
    return preds


@pytest.fixture
def convqa_golden_data():
    return [
        {
            "video_path": "v1.mp4",
            "task": "greeting",
            "category": "social",
            "questions": ["What did they say?", "How did they respond?"],
            "answers": ["Hello there", "Nice to meet you too"],
        },
        {
            "video_path": "v2.mp4",
            "task": "navigation",
            "category": "spatial",
            "questions": ["Where are they going?"],
            "answers": ["They are going to the park"],
        },
    ]


@pytest.fixture
def convqa_preds_perfect(convqa_golden_data):
    return [{"answers": g["answers"][:]} for g in convqa_golden_data]


def write_jsonl(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ---------------------------------------------------------------------------
# Tests: normalize_answer
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("A", "A"),
            ("B", "B"),
            ("C", "C"),
            ("D", "D"),
            ("a", "A"),
            ("b", "B"),
            ("c", "C"),
            ("d", "D"),
        ],
    )
    def test_single_letter(self, raw, expected):
        assert ev.normalize_answer(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("A.", "A"),
            ("B.", "B"),
            ("A:", "A"),
            ("C: something", "C"),
            ("A)", "A"),
            ("(B)", "B"),
            ("(C)", "C"),
            ("Option A", "A"),
            ("option B", "B"),
            ("Option C.", "C"),
            ("Answer: A", "A"),
            ("answer: B", "B"),
            ("Answer: D", "D"),
            ("The answer is A.", "A"),
            ("The answer is B", "B"),
            ("I think the answer is C.", "C"),
            ("  A  ", "A"),
            ("\nB\n", "B"),
        ],
    )
    def test_formatted_answers(self, raw, expected):
        assert ev.normalize_answer(raw) == expected

    def test_empty_string(self):
        assert ev.normalize_answer("") == ""

    def test_no_valid_answer(self):
        assert ev.normalize_answer("xyz") == ""
        assert ev.normalize_answer("123") == ""

    def test_embedded_letter(self):
        assert ev.normalize_answer("I choose B as my answer") == "B"

    def test_first_char_fallback(self):
        assert ev.normalize_answer("Definitely") == "D"


# ---------------------------------------------------------------------------
# Tests: _ngrams
# ---------------------------------------------------------------------------


class TestNgrams:
    def test_unigrams(self):
        tokens = ["the", "cat", "sat"]
        result = ev._ngrams(tokens, 1)
        assert result[("the",)] == 1
        assert result[("cat",)] == 1
        assert result[("sat",)] == 1

    def test_bigrams(self):
        tokens = ["the", "cat", "sat"]
        result = ev._ngrams(tokens, 2)
        assert result[("the", "cat")] == 1
        assert result[("cat", "sat")] == 1
        assert len(result) == 2

    def test_repeated_tokens(self):
        tokens = ["the", "the", "the"]
        result = ev._ngrams(tokens, 1)
        assert result[("the",)] == 3

    def test_empty(self):
        result = ev._ngrams([], 1)
        assert len(result) == 0

    def test_n_larger_than_sequence(self):
        tokens = ["a", "b"]
        result = ev._ngrams(tokens, 3)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: _sentence_bleu
# ---------------------------------------------------------------------------


class TestSentenceBleu:
    def test_perfect_match(self):
        ref = ["the", "cat", "sat", "on", "the", "mat"]
        hyp = ["the", "cat", "sat", "on", "the", "mat"]
        score = ev._sentence_bleu(ref, hyp)
        assert score > 0.99

    def test_empty_hypothesis(self):
        ref = ["the", "cat"]
        hyp = []
        assert ev._sentence_bleu(ref, hyp) == 0.0

    def test_partial_match(self):
        ref = ["the", "cat", "sat", "on", "the", "mat"]
        hyp = ["the", "cat", "is", "here"]
        score = ev._sentence_bleu(ref, hyp)
        assert score == pytest.approx(0.2868, abs=0.01)

    def test_no_overlap(self):
        ref = ["the", "cat", "sat", "on", "the", "mat"]
        hyp = ["xyz", "abc", "def", "ghi", "jkl", "mno"]
        score = ev._sentence_bleu(ref, hyp)
        assert score == pytest.approx(0.1858, abs=0.01)

    def test_brevity_penalty(self):
        ref = ["the", "cat", "sat", "on", "the", "mat"]
        hyp = ["the", "cat"]
        score_short = ev._sentence_bleu(ref, hyp)

        hyp_full = ["the", "cat", "sat", "on", "the", "mat"]
        score_full = ev._sentence_bleu(ref, hyp_full)

        assert score_full > score_short

    def test_symmetry_broken(self):
        ref = ["a", "b", "c"]
        hyp = ["a", "b", "c", "d", "e"]
        score1 = ev._sentence_bleu(ref, hyp)

        ref2 = ["a", "b", "c", "d", "e"]
        hyp2 = ["a", "b", "c"]
        score2 = ev._sentence_bleu(ref2, hyp2)
        assert score1 != score2


# ---------------------------------------------------------------------------
# Tests: _parse_judge_score
# ---------------------------------------------------------------------------


class TestParseJudgeScore:
    def test_exact_scores(self):
        assert ev._parse_judge_score("1.0") == 1.0
        assert ev._parse_judge_score("0.5") == 0.5
        assert ev._parse_judge_score("0.0") == 0.0

    def test_with_whitespace(self):
        assert ev._parse_judge_score("  1.0  ") == 1.0
        assert ev._parse_judge_score("\n0.5\n") == 0.5

    def test_with_trailing_period(self):
        assert ev._parse_judge_score("1.0.") == 1.0

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("0.8", 1.0),
            ("0.75", 1.0),
            ("0.4", 0.5),
            ("0.25", 0.5),
            ("0.1", 0.0),
            ("0.24", 0.0),
        ],
    )
    def test_threshold_mapping(self, raw, expected):
        assert ev._parse_judge_score(raw) == expected

    def test_no_number(self):
        assert ev._parse_judge_score("great answer") == 0.0

    def test_mixed_text_and_number(self):
        assert ev._parse_judge_score("Score: 1.0") == 1.0
        assert ev._parse_judge_score("I give it a 0.5 out of 1") == 0.5


# ---------------------------------------------------------------------------
# Tests: _build_judge_prompt
# ---------------------------------------------------------------------------


class TestBuildJudgePrompt:
    def test_prompt_structure(self):
        messages = ev._build_judge_prompt("What?", "blue", "red")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert "What?" in content
        assert "blue" in content
        assert "red" in content
        assert "Score:" in content

    def test_prompt_contains_rubric(self):
        messages = ev._build_judge_prompt("Q", "A", "P")
        content = messages[0]["content"]
        assert "1.0" in content
        assert "0.5" in content
        assert "0.0" in content


# ---------------------------------------------------------------------------
# Tests: load_jsonl
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_basic_load(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        write_jsonl(path, data)
        loaded = ev.load_jsonl(path)
        assert len(loaded) == 3
        assert loaded[0]["a"] == 1
        assert loaded[2]["c"] == 3

    def test_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.jsonl")
        with open(path, "w"):
            pass
        loaded = ev.load_jsonl(path)
        assert loaded == []

    def test_blank_lines_skipped(self, tmp_dir):
        path = os.path.join(tmp_dir, "blanks.jsonl")
        with open(path, "w") as f:
            f.write('{"a": 1}\n')
            f.write("\n")
            f.write('{"b": 2}\n')
            f.write("   \n")
        loaded = ev.load_jsonl(path)
        assert len(loaded) == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ev.load_jsonl("/nonexistent/path.jsonl")


# ---------------------------------------------------------------------------
# Tests: evaluate_longqa
# ---------------------------------------------------------------------------


class TestEvaluateLongqa:
    def test_perfect_accuracy(self, longqa_golden_data, longqa_preds_perfect):
        results = ev.evaluate_longqa(longqa_golden_data, longqa_preds_perfect)
        assert results["accuracy"] == 1.0
        assert results["correct"] == 4
        assert results["total"] == 4
        # category breakdown
        cat_acc = results["category_accuracy"]
        assert "color" in cat_acc
        assert "count" in cat_acc
        assert "time" in cat_acc
        assert cat_acc["color"] == 1.0
        # per-row results
        assert len(results["per_row"]) == 4
        for row in results["per_row"]:
            assert row["correct"] is True
            assert "video_path" in row
            assert "question" in row

    def test_half_accuracy(self, longqa_golden_data, longqa_preds_half):
        results = ev.evaluate_longqa(longqa_golden_data, longqa_preds_half)
        assert results["accuracy"] == 0.5
        assert results["correct"] == 2
        assert results["total"] == 4

    def test_empty_inputs(self):
        results = ev.evaluate_longqa([], [])
        assert results["accuracy"] == 0.0
        assert results["total"] == 0

    def test_verbose_answer_formats(self):
        golden = [
            {
                "video_path": "v1.mp4",
                "question": "Q",
                "mcq_answer": "B",
                "category": "",
            },
        ]
        preds = [{"mcq_answer": "The answer is B."}]
        results = ev.evaluate_longqa(golden, preds)
        assert results["accuracy"] == 1.0

    def test_missing_mcq_answer_in_pred(self):
        golden = [
            {"video_path": "v1.mp4", "question": "Q", "mcq_answer": "A", "category": ""}
        ]
        preds = [{}]
        results = ev.evaluate_longqa(golden, preds)
        assert results["accuracy"] == 0.0

    def test_length_mismatch_raises(self, longqa_golden_data):
        preds = [{"mcq_answer": "A"}]
        with pytest.raises(ValueError, match="must have the same number"):
            ev.evaluate_longqa(longqa_golden_data, preds)


# ---------------------------------------------------------------------------
# Tests: compute_bleu_scores
# ---------------------------------------------------------------------------


class TestComputeBleuScores:
    def test_perfect_scores(self, convqa_golden_data, convqa_preds_perfect):
        scores = ev.compute_bleu_scores(convqa_golden_data, convqa_preds_perfect)
        assert len(scores) == 2
        assert len(scores[0]) == 2
        assert len(scores[1]) == 1
        for turn_scores in scores:
            for s in turn_scores:
                assert s > 0.99

    def test_empty_prediction(self, convqa_golden_data):
        preds = [{"answers": ["", ""]}, {"answers": [""]}]
        scores = ev.compute_bleu_scores(convqa_golden_data, preds)
        for turn_scores in scores:
            for s in turn_scores:
                assert s == 0.0

    def test_missing_turns(self, convqa_golden_data):
        preds = [{"answers": ["Hello there"]}, {"answers": []}]
        scores = ev.compute_bleu_scores(convqa_golden_data, preds)
        assert len(scores[0]) == 2
        assert scores[0][0] > 0.5
        assert scores[0][1] == 0.0
        assert scores[1][0] == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_llm_judge_scores
# ---------------------------------------------------------------------------


class TestComputeLlmJudgeScores:
    def _make_mock_processor(self, decode_output="1.0"):
        from unittest.mock import MagicMock

        processor = MagicMock()

        class FakeInputs(dict):
            def to(self, device):
                return self

        processor.side_effect = lambda *a, **kw: FakeInputs(
            {"input_ids": MagicMock(shape=[1, 5])}
        )
        processor.apply_chat_template = MagicMock(return_value="mock_template")
        processor.decode = MagicMock(return_value=decode_output)
        return processor

    def _make_mock_model(self):
        from unittest.mock import MagicMock

        model = MagicMock()
        model.device = "cpu"
        model.generate.return_value = [MagicMock()]
        return model

    def test_empty_prediction_returns_zero(self):
        from unittest.mock import patch

        golden = [
            {"questions": ["Q1"], "answers": ["gold answer"]},
        ]
        preds = [{"answers": [""]}]
        mock_proc = self._make_mock_processor()
        mock_model = self._make_mock_model()
        with patch.object(
            ev, "_load_judge_model", return_value=(mock_proc, mock_model)
        ):
            scores = ev.compute_llm_judge_scores(golden, preds, "mock-model", 1)
        assert scores == [[0.0]]
        mock_model.generate.assert_not_called()

    def test_mocked_model_returns_expected_score(self):
        from unittest.mock import patch

        golden = [
            {"questions": ["Q1"], "answers": ["gold answer"]},
        ]
        preds = [{"answers": ["my prediction"]}]
        mock_proc = self._make_mock_processor(decode_output="1.0")
        mock_model = self._make_mock_model()
        with patch.object(
            ev, "_load_judge_model", return_value=(mock_proc, mock_model)
        ):
            scores = ev.compute_llm_judge_scores(golden, preds, "mock-model", 1)
        assert scores == [[1.0]]
        mock_model.generate.assert_called_once()

    def test_exception_during_generation_returns_zero(self):
        from unittest.mock import patch

        golden = [
            {"questions": ["Q1"], "answers": ["gold answer"]},
        ]
        preds = [{"answers": ["my prediction"]}]
        mock_proc = self._make_mock_processor()
        mock_model = self._make_mock_model()
        mock_proc.side_effect = RuntimeError("GPU OOM")
        with patch.object(
            ev, "_load_judge_model", return_value=(mock_proc, mock_model)
        ):
            scores = ev.compute_llm_judge_scores(golden, preds, "mock-model", 1)
        assert scores == [[0.0]]

    def test_missing_prediction_turns_returns_zero(self):
        from unittest.mock import patch

        golden = [
            {"questions": ["Q1", "Q2"], "answers": ["ans1", "ans2"]},
        ]
        preds = [{"answers": []}]
        mock_proc = self._make_mock_processor()
        mock_model = self._make_mock_model()
        with patch.object(
            ev, "_load_judge_model", return_value=(mock_proc, mock_model)
        ):
            scores = ev.compute_llm_judge_scores(golden, preds, "mock-model", 1)
        assert scores == [[0.0, 0.0]]


# ---------------------------------------------------------------------------
# Tests: evaluate_convqa
# ---------------------------------------------------------------------------


class TestEvaluateConvqa:
    def test_bleu_perfect(self, convqa_golden_data, convqa_preds_perfect):
        results = ev.evaluate_convqa(
            convqa_golden_data,
            convqa_preds_perfect,
            run_bleu=True,
            run_llm_judge=False,
        )
        assert "bleu" in results
        assert results["bleu"] > 0.99
        assert results["total_conversations"] == 2
        assert results["total_turns"] == 3
        # category breakdown
        assert "category_scores" in results
        cats = results["category_scores"]
        assert "social" in cats
        assert "spatial" in cats
        assert cats["social"]["bleu"] > 0.99
        # per-row structure
        per_row = results["per_row"]
        assert len(per_row) == 2
        for row in per_row:
            assert "video_path" in row
            assert "bleu_avg" in row
            assert "bleu_per_turn" in row
            assert "num_turns" in row

    def test_no_bleu_no_judge(self, convqa_golden_data, convqa_preds_perfect):
        results = ev.evaluate_convqa(
            convqa_golden_data,
            convqa_preds_perfect,
            run_bleu=False,
            run_llm_judge=False,
        )
        assert "bleu" not in results
        assert "llm_judge" not in results

    def test_empty_inputs(self):
        results = ev.evaluate_convqa([], [], run_bleu=True, run_llm_judge=False)
        assert results["total_conversations"] == 0

    def test_length_mismatch_raises(self, convqa_golden_data):
        preds = [convqa_golden_data[0]]
        with pytest.raises(ValueError, match="must have the same number"):
            ev.evaluate_convqa(convqa_golden_data, preds)


# ---------------------------------------------------------------------------
# Tests: _resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_absolute_path_unchanged(self):
        assert ev._resolve_path("/tmp/test.jsonl") == "/tmp/test.jsonl"

    def test_relative_path_resolved(self):
        result = ev._resolve_path("data/test.jsonl")
        assert os.path.isabs(result)
        assert result.endswith("data/test.jsonl")


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    @pytest.mark.parametrize("task", ["longqa", "convqa", "all"])
    def test_valid_task(self, task):
        parser = ev._build_parser()
        args = parser.parse_args(["--task", task])
        assert args.task == task

    def test_task_required(self):
        parser = ev._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_invalid_task(self):
        parser = ev._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--task", "invalid"])

    def test_golden_and_predictions(self):
        parser = ev._build_parser()
        args = parser.parse_args(
            [
                "--task",
                "longqa",
                "--golden",
                "my_golden.jsonl",
                "--predictions",
                "my_preds.jsonl",
            ]
        )
        assert args.golden == "my_golden.jsonl"
        assert args.predictions == "my_preds.jsonl"

    def test_multi_task_files(self):
        parser = ev._build_parser()
        args = parser.parse_args(
            [
                "--task",
                "all",
                "--golden-longqa",
                "lq_gold.jsonl",
                "--predictions-longqa",
                "lq_pred.jsonl",
                "--golden-convqa",
                "cq_gold.jsonl",
                "--predictions-convqa",
                "cq_pred.jsonl",
            ]
        )
        assert args.golden_longqa == "lq_gold.jsonl"
        assert args.predictions_longqa == "lq_pred.jsonl"
        assert args.golden_convqa == "cq_gold.jsonl"
        assert args.predictions_convqa == "cq_pred.jsonl"

    def test_llm_judge_flag(self):
        parser = ev._build_parser()
        args = parser.parse_args(["--task", "convqa", "--llm-judge"])
        assert args.llm_judge is True

    def test_no_llm_judge_flag(self):
        parser = ev._build_parser()
        args = parser.parse_args(["--task", "convqa", "--no-llm-judge"])
        assert args.llm_judge is False

    def test_judge_model_override(self):
        parser = ev._build_parser()
        args = parser.parse_args(
            [
                "--task",
                "convqa",
                "--judge-model",
                "my-model/v1",
            ]
        )
        assert args.llm_judge_model == "my-model/v1"

    def test_default_judge_model(self):
        parser = ev._build_parser()
        args = parser.parse_args(["--task", "convqa"])
        assert args.llm_judge_model == ev.DEFAULT_JUDGE_MODEL

    def test_output_path(self):
        parser = ev._build_parser()
        args = parser.parse_args(
            [
                "--task",
                "longqa",
                "--output",
                "/tmp/results.json",
            ]
        )
        assert args.output == "/tmp/results.json"

    def test_judge_batch_size(self):
        parser = ev._build_parser()
        args = parser.parse_args(
            [
                "--task",
                "convqa",
                "--judge-batch-size",
                "8",
            ]
        )
        assert args.judge_batch_size == 8


# ---------------------------------------------------------------------------
# Tests: E2E integration (LongQA)
# ---------------------------------------------------------------------------


class TestE2ELongqa:
    def test_full_pipeline(self, tmp_dir, longqa_golden_data, longqa_preds_perfect):
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        write_jsonl(golden_path, longqa_golden_data)
        write_jsonl(preds_path, longqa_preds_perfect)

        ev._run_longqa(golden_path, preds_path, output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        assert results["accuracy"] == 1.0
        assert results["total"] == 4

    @pytest.mark.parametrize(
        "indices",
        [[0, 3], [3, 0]],
        ids=["non_contiguous", "reversed"],
    )
    def test_subset_evaluation(self, tmp_dir, longqa_golden_data, indices):
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        # Use non-contiguous/reversed subset to verify dict-based
        # video_path filtering rather than naive positional truncation
        subset = [longqa_golden_data[i] for i in indices]
        write_jsonl(golden_path, longqa_golden_data)
        write_jsonl(preds_path, subset)

        ev._run_longqa(golden_path, preds_path, output_path)
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        assert results["total"] == 2
        # Verify correct video_path filtering produces perfect accuracy,
        # not just naive truncation of first N entries
        assert results["accuracy"] == 1.0

    def test_subset_unknown_video_path(self, tmp_dir, longqa_golden_data):
        """Prediction with video_path absent from golden is filtered out.

        The subset filtering keeps only predictions whose (video_path,
        question) key exists in golden.  The unknown prediction is dropped,
        leaving only the matched prediction for evaluation.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        # One valid prediction (with question for composite key) + one unknown
        preds = [
            {
                "video_path": longqa_golden_data[0]["video_path"],
                "question": longqa_golden_data[0]["question"],
                "mcq_answer": longqa_golden_data[0]["mcq_answer"],
            },
            {"video_path": "unknown.mp4", "question": "What?", "mcq_answer": "A"},
        ]
        write_jsonl(golden_path, longqa_golden_data)
        write_jsonl(preds_path, preds)

        # Filtering drops the unknown prediction and evaluates the 1 match.
        ev._run_longqa(golden_path, preds_path, output_path)
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        assert results["total"] == 1
        assert results["accuracy"] == 1.0

    def test_subset_missing_video_path(self, tmp_dir, longqa_golden_data, caplog):
        """Prediction missing video_path logs a warning and is filtered out.

        The warning about incomplete filtering is logged, then the prediction
        without video_path is dropped (its key is (None, ...)) and only the
        matched prediction is evaluated.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        # One valid prediction (with question for composite key) + one missing video_path
        preds = [
            {
                "video_path": longqa_golden_data[0]["video_path"],
                "question": longqa_golden_data[0]["question"],
                "mcq_answer": longqa_golden_data[0]["mcq_answer"],
            },
            {"mcq_answer": "A"},
        ]
        write_jsonl(golden_path, longqa_golden_data)
        write_jsonl(preds_path, preds)

        import logging

        with caplog.at_level(logging.WARNING):
            ev._run_longqa(golden_path, preds_path, output_path)
        assert "missing video_path" in caplog.text
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        assert results["total"] == 1
        assert results["accuracy"] == 1.0

    def test_subset_preds_outnumber_golden(self, tmp_dir, longqa_golden_data):
        """When predictions outnumber golden, extra unknown preds are dropped.

        The subset filtering matches predictions against golden by the
        (video_path, question) composite key.  Extra predictions whose keys
        are absent from golden are silently dropped, and only the matched
        subset is evaluated.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        # Build predictions that cover ALL golden entries plus extras
        preds = [
            {
                "video_path": g["video_path"],
                "question": g["question"],
                "mcq_answer": g["mcq_answer"],
            }
            for g in longqa_golden_data
        ]
        # Add extra predictions with unknown video_paths
        preds.extend(
            [
                {
                    "video_path": "extra_unknown_1.mp4",
                    "question": "Extra Q1?",
                    "mcq_answer": "A",
                },
                {
                    "video_path": "extra_unknown_2.mp4",
                    "question": "Extra Q2?",
                    "mcq_answer": "B",
                },
            ]
        )
        assert len(preds) > len(longqa_golden_data)

        write_jsonl(golden_path, longqa_golden_data)
        write_jsonl(preds_path, preds)

        ev._run_longqa(golden_path, preds_path, output_path)
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        # Only the 4 matched predictions are evaluated; extras are dropped.
        assert results["total"] == len(longqa_golden_data)
        assert results["accuracy"] == 1.0

    def test_subset_preds_fewer_than_golden_imperfect(
        self, tmp_dir, longqa_golden_data
    ):
        """Subset evaluation with imperfect predictions correctly reports accuracy.

        Creates golden with 4 entries but predictions with only 2 matching
        entries, one correct and one wrong.  Verifies that subset filtering
        matches by composite key AND accuracy reflects the imperfect predictions.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        # 2 predictions: first correct, second deliberately wrong
        g0 = longqa_golden_data[0]
        g1 = longqa_golden_data[1]
        wrong_answer = "A" if g1["mcq_answer"] != "A" else "B"
        preds = [
            {
                "video_path": g0["video_path"],
                "question": g0["question"],
                "mcq_answer": g0["mcq_answer"],
            },
            {
                "video_path": g1["video_path"],
                "question": g1["question"],
                "mcq_answer": wrong_answer,
            },
        ]
        assert len(preds) < len(longqa_golden_data)

        write_jsonl(golden_path, longqa_golden_data)
        write_jsonl(preds_path, preds)

        ev._run_longqa(golden_path, preds_path, output_path)
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        # Only the 2 matched predictions are evaluated, 1/2 correct.
        assert results["total"] == 2
        assert results["accuracy"] == 0.5

    # ---------------------------------------------------------------------------

    def test_subset_duplicate_pred_keys(self, tmp_dir, longqa_golden_data, caplog):
        """Duplicate composite keys in predictions are deduplicated (first wins).

        When two predictions share the same (video_path, question) key, only
        the first is kept.  This prevents a single golden entry from being
        counted multiple times, which would inflate evaluation metrics.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        g0 = longqa_golden_data[0]
        # Two predictions with the same key: first correct, second wrong
        wrong_answer = "A" if g0["mcq_answer"] != "A" else "B"
        preds = [
            {
                "video_path": g0["video_path"],
                "question": g0["question"],
                "mcq_answer": g0["mcq_answer"],
            },
            {
                "video_path": g0["video_path"],
                "question": g0["question"],
                "mcq_answer": wrong_answer,
            },
        ]

        write_jsonl(golden_path, longqa_golden_data)
        write_jsonl(preds_path, preds)

        with caplog.at_level(logging.WARNING):
            ev._run_longqa(golden_path, preds_path, output_path)

        assert "duplicate prediction key" in caplog.text
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        # Only the first prediction (correct) is kept
        assert results["total"] == 1
        assert results["accuracy"] == 1.0


# Tests: E2E integration (ConvQA)
# ---------------------------------------------------------------------------


class TestE2EConvqa:
    def test_full_pipeline_bleu_only(
        self, tmp_dir, convqa_golden_data, convqa_preds_perfect
    ):
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        write_jsonl(golden_path, convqa_golden_data)
        write_jsonl(preds_path, convqa_preds_perfect)

        ev._run_convqa(
            golden_path,
            preds_path,
            output_path,
            run_llm_judge=False,
            judge_model="",
            judge_batch_size=1,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        assert "bleu" in results
        assert results["bleu"] > 0.99

    def test_subset_evaluation(self, tmp_dir, convqa_golden_data):
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        # Use second entry (v2.mp4) to verify video_path-based filtering
        # rather than naive positional truncation
        write_jsonl(golden_path, convqa_golden_data)
        write_jsonl(preds_path, convqa_golden_data[1:])

        ev._run_convqa(
            golden_path,
            preds_path,
            output_path,
            run_llm_judge=False,
            judge_model="",
            judge_batch_size=1,
        )
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        assert results["total_conversations"] == 1
        # Verify correct matching produces high BLEU (not mismatched pairs)
        assert results["bleu"] > 0.99

    def test_subset_unknown_video_path(self, tmp_dir, convqa_golden_data):
        """Prediction with video_path absent from golden raises ValueError.

        The subset filtering keeps only predictions whose (video_path, task)
        key exists in golden.  When all predictions are dropped, zero matched
        pairs remain and _run_convqa raises ValueError.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        preds = [
            {
                "video_path": "unknown.mp4",
                "task": "unknown_task",
                "questions": ["What?"],
                "answers": ["Nothing"],
            },
        ]
        write_jsonl(golden_path, convqa_golden_data)
        write_jsonl(preds_path, preds)

        with pytest.raises(ValueError, match="zero predictions matched"):
            ev._run_convqa(
                golden_path,
                preds_path,
                output_path,
                run_llm_judge=False,
                judge_model="",
                judge_batch_size=1,
            )

    def test_subset_missing_video_path(self, tmp_dir, convqa_golden_data, caplog):
        """Prediction missing video_path logs a warning then raises ValueError.

        The warning about missing video_path is logged, then the prediction
        is dropped (its key is (None, ...)).  With zero matched pairs,
        _run_convqa raises ValueError.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        preds = [
            {"questions": ["What?"], "answers": ["Nothing"]},
        ]
        write_jsonl(golden_path, convqa_golden_data)
        write_jsonl(preds_path, preds)

        import logging

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="zero predictions matched"):
                ev._run_convqa(
                    golden_path,
                    preds_path,
                    output_path,
                    run_llm_judge=False,
                    judge_model="",
                    judge_batch_size=1,
                )
        assert "missing video_path" in caplog.text

    def test_subset_preds_outnumber_golden(self, tmp_dir, convqa_golden_data):
        """When predictions outnumber golden, extra unknown preds are dropped.

        The subset filtering matches predictions against golden by the
        (video_path, task) composite key.  Extra predictions whose keys
        are absent from golden are silently dropped, and only the matched
        subset is evaluated.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        # Build predictions that cover ALL golden entries plus extras
        preds = [
            {
                "video_path": g["video_path"],
                "task": g["task"],
                "questions": g["questions"][:],
                "answers": g["answers"][:],
            }
            for g in convqa_golden_data
        ]
        # Add extra predictions with unknown video_paths
        preds.extend(
            [
                {
                    "video_path": "extra_unknown_1.mp4",
                    "task": "unknown_task_1",
                    "questions": ["What happened?"],
                    "answers": ["Something happened"],
                },
                {
                    "video_path": "extra_unknown_2.mp4",
                    "task": "unknown_task_2",
                    "questions": ["Who is there?"],
                    "answers": ["Nobody"],
                },
            ]
        )
        assert len(preds) > len(convqa_golden_data)

        write_jsonl(golden_path, convqa_golden_data)
        write_jsonl(preds_path, preds)

        ev._run_convqa(
            golden_path,
            preds_path,
            output_path,
            run_llm_judge=False,
            judge_model="",
            judge_batch_size=1,
        )
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        # Only the 2 matched predictions are evaluated; extras are dropped.
        assert results["total_conversations"] == len(convqa_golden_data)
        assert results["bleu"] > 0.99

    def test_subset_preds_fewer_than_golden_imperfect(
        self, tmp_dir, convqa_golden_data
    ):
        """Subset evaluation with imperfect predictions correctly reports BLEU.

        Creates golden with 2 entries but predictions with only 1 matching
        entry whose answers are deliberately wrong.  Verifies that subset
        filtering matches by (video_path, task) composite key AND BLEU
        reflects the imperfect predictions (low but non-zero).
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        # 1 prediction with deliberately wrong answers for low BLEU
        g0 = convqa_golden_data[0]
        wrong_answers = ["completely wrong unrelated response" for _ in g0["answers"]]
        preds = [
            {
                "video_path": g0["video_path"],
                "task": g0["task"],
                "questions": g0["questions"][:],
                "answers": wrong_answers,
            }
        ]
        assert len(preds) < len(convqa_golden_data)

        write_jsonl(golden_path, convqa_golden_data)
        write_jsonl(preds_path, preds)

        ev._run_convqa(
            golden_path,
            preds_path,
            output_path,
            run_llm_judge=False,
            judge_model="",
            judge_batch_size=1,
        )
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        # Only the 1 matched prediction is evaluated, with imperfect BLEU.
        assert results["total_conversations"] == 1
        assert results["bleu"] < 0.5
        assert results["bleu"] > 0.0

    # ---------------------------------------------------------------------------

    def test_subset_duplicate_pred_keys(self, tmp_dir, convqa_golden_data, caplog):
        """Duplicate composite keys in predictions are deduplicated (first wins).

        When two predictions share the same (video_path, task) key, only
        the first is kept.  This prevents a single golden entry from being
        counted multiple times, which would inflate evaluation metrics.

        We send 3 predictions (2 for g0 with same key + 1 for g1) so that
        ``len(preds) != len(golden)`` triggers the ``_filter_subset`` path.
        After dedup the duplicate is dropped, leaving 2 matched predictions.
        """
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "results.json")

        g0 = convqa_golden_data[0]
        g1 = convqa_golden_data[1]
        # 3 predictions: g0 correct, g0 duplicate (wrong), g1 correct
        preds = [
            {
                "video_path": g0["video_path"],
                "task": g0["task"],
                "questions": g0["questions"][:],
                "answers": g0["answers"][:],
            },
            {
                "video_path": g0["video_path"],
                "task": g0["task"],
                "questions": g0["questions"][:],
                "answers": ["completely wrong answer" for _ in g0["answers"]],
            },
            {
                "video_path": g1["video_path"],
                "task": g1["task"],
                "questions": g1["questions"][:],
                "answers": g1["answers"][:],
            },
        ]

        write_jsonl(golden_path, convqa_golden_data)
        write_jsonl(preds_path, preds)

        with caplog.at_level(logging.WARNING):
            ev._run_convqa(
                golden_path,
                preds_path,
                output_path,
                run_llm_judge=False,
                judge_model="",
                judge_batch_size=1,
            )

        assert "duplicate prediction key" in caplog.text
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        # Duplicate dropped: 2 matched predictions (g0 correct + g1 correct)
        assert results["total_conversations"] == 2
        assert results["bleu"] > 0.99


# Tests: E2E CLI subprocess
# ---------------------------------------------------------------------------


class TestE2ECLI:
    def test_cli_longqa(self, tmp_dir, longqa_golden_data, longqa_preds_perfect):
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "cli_results.json")

        write_jsonl(golden_path, longqa_golden_data)
        write_jsonl(preds_path, longqa_preds_perfect)

        result = subprocess.run(
            [
                sys.executable,
                EVAL_SCRIPT,
                "--task",
                "longqa",
                "--golden",
                golden_path,
                "--predictions",
                preds_path,
                "--output",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        assert results["accuracy"] == 1.0

    def test_cli_convqa_bleu(self, tmp_dir, convqa_golden_data, convqa_preds_perfect):
        golden_path = os.path.join(tmp_dir, "golden.jsonl")
        preds_path = os.path.join(tmp_dir, "preds.jsonl")
        output_path = os.path.join(tmp_dir, "cli_results.json")

        write_jsonl(golden_path, convqa_golden_data)
        write_jsonl(preds_path, convqa_preds_perfect)

        result = subprocess.run(
            [
                sys.executable,
                EVAL_SCRIPT,
                "--task",
                "convqa",
                "--golden",
                golden_path,
                "--predictions",
                preds_path,
                "--output",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(output_path)
        with open(output_path) as f:
            results = json.load(f)
        assert "bleu" in results
        assert results["bleu"] > 0.99

    def test_cli_missing_task(self):
        result = subprocess.run(
            [sys.executable, EVAL_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "--task" in result.stderr

    def test_cli_invalid_task(self):
        result = subprocess.run(
            [sys.executable, EVAL_SCRIPT, "--task", "invalid"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "invalid" in result.stderr


# ---------------------------------------------------------------------------
# Tests: Output format validation
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_results_json_serializable(self, longqa_golden_data, longqa_preds_perfect):
        results = ev.evaluate_longqa(longqa_golden_data, longqa_preds_perfect)
        serialized = json.dumps(results)
        roundtrip = json.loads(serialized)
        assert roundtrip == results


# ---------------------------------------------------------------------------
# Tests: Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sample(self):
        golden = [
            {"video_path": "v.mp4", "question": "Q", "mcq_answer": "A", "category": ""}
        ]
        preds = [{"mcq_answer": "A"}]
        results = ev.evaluate_longqa(golden, preds)
        assert results["accuracy"] == 1.0
        assert results["total"] == 1

    def test_convqa_partial_predictions(self, convqa_golden_data):
        preds = [
            {"answers": ["Hello"]},
            {"answers": []},
        ]
        results = ev.evaluate_convqa(
            convqa_golden_data,
            preds,
            run_bleu=True,
            run_llm_judge=False,
        )
        assert "bleu" in results
        assert 0.0 < results["bleu"] < 1.0, (
            "Partial predictions should yield a BLEU score between 0 and 1 (not perfect)"
        )
        assert results["total_turns"] == 3

    def test_convqa_extra_prediction_turns_ignored(self, convqa_golden_data):
        preds = [
            {"answers": ["Hello there", "Nice to meet you too", "Extra turn"]},
            {"answers": ["They are going to the park", "Extra"]},
        ]
        scores = ev.compute_bleu_scores(convqa_golden_data, preds)
        assert len(scores[0]) == 2
        assert len(scores[1]) == 1
        assert scores[0][0] > 0.99
        assert scores[0][1] > 0.99
        assert scores[1][0] > 0.99

    def test_longqa_category_none(self):
        golden = [{"video_path": "v.mp4", "question": "Q", "mcq_answer": "A"}]
        preds = [{"mcq_answer": "A"}]
        results = ev.evaluate_longqa(golden, preds)
        assert results["accuracy"] == 1.0

    def test_unicode_answers(self):
        golden = [
            {"video_path": "v.mp4", "question": "Q", "mcq_answer": "A", "category": ""}
        ]
        preds = [{"mcq_answer": "A — correct"}]
        results = ev.evaluate_longqa(golden, preds)
        assert results["accuracy"] == 1.0

    def test_multiline_answer(self):
        raw = "I think\nthe answer\nis A."
        assert ev.normalize_answer(raw) == "A"

    def test_convqa_missing_answers_key(self, convqa_golden_data):
        preds = [{}, {}]
        scores = ev.compute_bleu_scores(convqa_golden_data, preds)
        for turn_scores in scores:
            for s in turn_scores:
                assert s == 0.0

    @pytest.mark.parametrize("wrong_answer", ["Z", "E", "1", ""])
    def test_longqa_all_wrong_parametrized(self, longqa_golden_data, wrong_answer):
        preds = [{"mcq_answer": wrong_answer} for _ in longqa_golden_data]
        results = ev.evaluate_longqa(longqa_golden_data, preds)
        assert results["accuracy"] == 0.0
        assert results["correct"] == 0
        assert results["total"] == len(longqa_golden_data)


# ---------------------------------------------------------------------------
# model.py tests: setup_gpus + generate_batch decode slicing
# ---------------------------------------------------------------------------

import model as mdl


# ---------------------------------------------------------------------------
# Tests: VideoQAModel context manager protocol
# ---------------------------------------------------------------------------


class TestVideoQAModelContextManager:
    def test_hf_model_works_as_context_manager(self):
        class DummyModel(mdl.VideoQAModel):
            def generate(self, frames, messages, max_new_tokens=256):
                return "test"

        with DummyModel() as model:
            assert model.generate([], []) == "test"

    def test_base_class_exit_returns_false(self):
        class DummyModel(mdl.VideoQAModel):
            def generate(self, frames, messages, max_new_tokens=256):
                return "test"

        model = DummyModel()
        assert model.__exit__(None, None, None) is False

    def test_enter_returns_self(self):
        class DummyModel(mdl.VideoQAModel):
            def generate(self, frames, messages, max_new_tokens=256):
                return "test"

        model = DummyModel()
        assert model.__enter__() is model


# ---------------------------------------------------------------------------
# Tests: find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_valid_port(self):
        port = mdl.find_free_port()
        assert 1024 < port < 65536


# ---------------------------------------------------------------------------
# Tests: VLLMModel
# ---------------------------------------------------------------------------


class TestVLLMModel:
    def test_init_stores_config(self):
        model = mdl.VLLMModel("test-model", tp_size=2, concurrency=8)
        assert model.model_id == "test-model"
        assert model.tp_size == 2
        assert model.concurrency == 8
        assert model._proc is None
        assert model._port is None

    def test_init_defaults(self):
        model = mdl.VLLMModel("test-model")
        assert model.tp_size == 1
        assert model.concurrency == 16

    @staticmethod
    def _make_fake_image():
        """Create a mock image with a .save() method that writes valid JPEG-like bytes."""
        fake_img = unittest.mock.MagicMock()

        def save_side_effect(buf, format="JPEG"):
            buf.write(b"\xff\xd8\xff\xe0fake_jpeg_data\xff\xd9")

        fake_img.save = save_side_effect
        return fake_img

    @staticmethod
    def _patch_urlopen(model, response_content, port=9999):
        """Set up urlopen mock for VLLMModel.generate() tests."""
        mock_response = unittest.mock.MagicMock()
        mock_response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": response_content}}]}
        ).encode()
        mock_response.__enter__ = unittest.mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = unittest.mock.MagicMock(return_value=False)
        model._port = port
        patcher = unittest.mock.patch(
            "urllib.request.urlopen", return_value=mock_response
        )
        return patcher

    def test_generate_builds_correct_payload(self):
        model = mdl.VLLMModel("test-model", tp_size=1)
        frames = [self._make_fake_image()]
        messages = [{"role": "user", "content": "What is this?"}]

        with self._patch_urlopen(model, "A red image") as mock_urlopen:
            result = model.generate(frames, messages)
            assert result == "A red image"

            req = mock_urlopen.call_args[0][0]
            payload = json.loads(req.data)
            assert payload["model"] == "test-model"
            assert payload["temperature"] == 0.0
            content = payload["messages"][0]["content"]
            assert content[0]["type"] == "image_url"
            assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            assert content[-1]["type"] == "text"
            assert content[-1]["text"] == "What is this?"

    def test_generate_no_frames(self):
        model = mdl.VLLMModel("test-model", tp_size=1)
        messages = [{"role": "user", "content": "Hello"}]

        with self._patch_urlopen(model, "Hi") as mock_urlopen:
            result = model.generate([], messages)
            assert result == "Hi"

            req = mock_urlopen.call_args[0][0]
            payload = json.loads(req.data)
            assert payload["messages"][0] == {"role": "user", "content": "Hello"}

    def test_generate_batch_preserves_order(self):
        model = mdl.VLLMModel("test-model", tp_size=1, concurrency=4)

        def mock_generate(frames, messages, max_new_tokens=4096):
            # Derive response from input content so we can verify ordering
            content = messages[0]["content"] if messages else "empty"
            return f"answer_for_{content}"

        model.generate = mock_generate
        batch_frames = [[], [], []]
        batch_messages = [
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q2"}],
            [{"role": "user", "content": "q3"}],
        ]

        results = model.generate_batch(batch_frames, batch_messages)
        assert len(results) == 3
        # Verify each result maps to the correct input position
        assert results[0] == "answer_for_q1"
        assert results[1] == "answer_for_q2"
        assert results[2] == "answer_for_q3"

    def test_generate_batch_raises_on_partial_failure(self):
        """Verify generate_batch raises RuntimeError on any request failure."""
        model = mdl.VLLMModel("test-model", tp_size=1, concurrency=4)

        def mock_generate(frames, messages, max_new_tokens=4096):
            content = messages[0]["content"] if messages else "empty"
            if content == "q2":
                raise RuntimeError("Simulated HTTP timeout")
            return f"answer_for_{content}"

        model.generate = mock_generate
        batch_frames = [[], [], []]
        batch_messages = [
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q2"}],
            [{"role": "user", "content": "q3"}],
        ]

        with pytest.raises(RuntimeError, match="1 / 3 vLLM requests failed"):
            model.generate_batch(batch_frames, batch_messages)

    def test_generate_batch_raises_when_all_requests_fail(self):
        """When every request fails, generate_batch raises RuntimeError."""
        model = mdl.VLLMModel("test-model", tp_size=1, concurrency=4)
        model.generate = unittest.mock.MagicMock(side_effect=RuntimeError("fail"))
        batch_frames = [[], []]
        batch_messages = [
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q2"}],
        ]
        with pytest.raises(RuntimeError, match="2 / 2 vLLM requests failed"):
            model.generate_batch(batch_frames, batch_messages)

    def test_generate_batch_raises_on_length_mismatch(self):
        model = mdl.VLLMModel("test-model", tp_size=1)
        with pytest.raises(ValueError, match="must have the same length"):
            model.generate_batch([[]], [], max_new_tokens=256)

    def test_kill_server_no_proc_is_noop(self):
        """Verify _kill_server is safe to call when no server process exists.

        This guards against crashes during cleanup when the server was never
        started (e.g., __exit__ called after __enter__ failed).
        """
        model = mdl.VLLMModel("test-model")
        model._proc = None
        model._kill_server()  # should not raise
        assert model._proc is None, "_proc should remain None after noop kill"

    def test_kill_server_with_proc_uses_killpg(self):
        """Verify _kill_server sends SIGTERM to the process group."""
        import signal

        model = mdl.VLLMModel("test-model")
        mock_proc = unittest.mock.MagicMock()
        mock_proc.pid = 99999
        mock_proc.wait.return_value = None
        model._proc = mock_proc
        model._log = None

        with (
            unittest.mock.patch("os.getpgid", return_value=99999) as mock_getpgid,
            unittest.mock.patch("os.killpg") as mock_killpg,
        ):
            model._kill_server()

        mock_getpgid.assert_called_once_with(99999)
        mock_killpg.assert_called_once_with(99999, signal.SIGTERM)
        assert model._proc is None

    def test_exit_returns_false(self):
        model = mdl.VLLMModel("test-model")
        model._proc = None
        assert model.__exit__(None, None, None) is False

    def test_multi_turn_images_only_in_first_user_message(self):
        model = mdl.VLLMModel("test-model", tp_size=1)
        frames = [self._make_fake_image()]
        messages = [
            {"role": "user", "content": "What is this?"},
            {"role": "assistant", "content": "A blue square."},
            {"role": "user", "content": "What color?"},
        ]

        with self._patch_urlopen(model, "Blue") as mock_urlopen:
            result = model.generate(frames, messages)
            assert result == "Blue"

            req = mock_urlopen.call_args[0][0]
            payload = json.loads(req.data)
            msgs = payload["messages"]
            # First user message has images
            assert isinstance(msgs[0]["content"], list)
            assert msgs[0]["content"][0]["type"] == "image_url"
            # Second message (assistant) is plain
            assert msgs[1]["content"] == "A blue square."
            # Third message (user) is plain — no images
            assert msgs[2]["content"] == "What color?"

    def test_generate_raises_on_vllm_error_response(self):
        """vLLM response with 'error' key raises RuntimeError."""
        model = mdl.VLLMModel("test-model", tp_size=1)
        model._port = 9999

        error_response = unittest.mock.MagicMock()
        error_response.read.return_value = json.dumps(
            {"error": "model overloaded"}
        ).encode()
        error_response.__enter__ = unittest.mock.MagicMock(return_value=error_response)
        error_response.__exit__ = unittest.mock.MagicMock(return_value=False)

        with unittest.mock.patch("urllib.request.urlopen", return_value=error_response):
            with pytest.raises(RuntimeError, match="vLLM returned error"):
                model.generate([], [{"role": "user", "content": "Hi"}])

    def test_generate_raises_on_unexpected_structure(self):
        """vLLM response missing choices raises RuntimeError."""
        model = mdl.VLLMModel("test-model", tp_size=1)
        model._port = 9999

        bad_response = unittest.mock.MagicMock()
        bad_response.read.return_value = json.dumps({"choices": []}).encode()
        bad_response.__enter__ = unittest.mock.MagicMock(return_value=bad_response)
        bad_response.__exit__ = unittest.mock.MagicMock(return_value=False)

        with unittest.mock.patch("urllib.request.urlopen", return_value=bad_response):
            with pytest.raises(RuntimeError, match="Unexpected vLLM response"):
                model.generate([], [{"role": "user", "content": "Hi"}])


# ---------------------------------------------------------------------------
# Tests: create_model backend routing
# ---------------------------------------------------------------------------


class TestCreateModelBackend:
    def test_create_model_vllm_returns_vllm_model(self):
        model = mdl.create_model("llama4", backend="vllm")
        assert isinstance(model, mdl.VLLMModel)
        assert model.model_id == mdl.DEFAULT_MODEL_IDS["llama4"]
        assert model.tp_size == mdl.DEFAULT_TP_SIZES["llama4"]

    def test_create_model_vllm_custom_tp(self):
        model = mdl.create_model("qwen", backend="vllm", tp_size=2, concurrency=32)
        assert isinstance(model, mdl.VLLMModel)
        assert model.tp_size == 2
        assert model.concurrency == 32

    def test_create_model_vllm_custom_model_id(self):
        model = mdl.create_model("qwen", model_id="/path/to/model", backend="vllm")
        assert isinstance(model, mdl.VLLMModel)
        assert model.model_id == "/path/to/model"

    def test_create_model_hf_dispatches_correctly(self):
        """Verify create_model(backend='hf') dispatches to the correct HF class."""
        with unittest.mock.patch.object(
            mdl.Qwen2VLModel, "__init__", return_value=None
        ):
            model = mdl.create_model("qwen", backend="hf")
            assert isinstance(model, mdl.Qwen2VLModel)

    def test_create_model_default_backend_is_hf(self):
        """Without backend param, create_model should return HF model instance."""
        with unittest.mock.patch.object(
            mdl.Qwen2VLModel, "__init__", return_value=None
        ):
            model = mdl.create_model("qwen")
            assert isinstance(model, mdl.Qwen2VLModel)

    def test_create_model_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            mdl.create_model("unknown", backend="vllm")

    def test_create_model_hf_backend_validates_registry(self):
        with pytest.raises(ValueError, match="HuggingFace backend requires"):
            mdl.create_model("unknown", backend="hf")


# ---------------------------------------------------------------------------
# Tests: DEFAULT_TP_SIZES constant
# ---------------------------------------------------------------------------


class TestDefaultTPSizes:
    @pytest.mark.parametrize(
        "model_type, expected_tp",
        [("llama4", 8), ("qwen", 1)],
    )
    def test_tp_size(self, model_type, expected_tp):
        assert mdl.DEFAULT_TP_SIZES[model_type] == expected_tp

    def test_matches_gpu_counts(self):
        for model_type in mdl.DEFAULT_TP_SIZES:
            assert model_type in mdl.DEFAULT_GPU_COUNTS


# ---------------------------------------------------------------------------
# Tests: CLI vLLM flags
# ---------------------------------------------------------------------------


class TestCLIVLLMFlags:
    @pytest.mark.parametrize(
        "cli_args, attr, expected",
        [
            (["--backend", "vllm"], "backend", "vllm"),
            (["--tp", "4"], "tp", 4),
            (["--concurrency", "32"], "concurrency", 32),
        ],
    )
    def test_vllm_flag_parsed(self, cli_args, attr, expected):
        parser = ev._build_parser()
        args = parser.parse_args(["--task", "longqa"] + cli_args)
        assert getattr(args, attr) == expected

    @pytest.mark.parametrize(
        "attr, expected",
        [
            ("backend", "hf"),
            ("concurrency", 16),
        ],
    )
    def test_vllm_flag_default(self, attr, expected):
        parser = ev._build_parser()
        args = parser.parse_args(["--task", "longqa"])
        assert getattr(args, attr) == expected


# ---------------------------------------------------------------------------
# Tests: VLLMModel _start_server command construction
# ---------------------------------------------------------------------------


class TestVLLMStartServer:
    """Verify _start_server constructs the correct subprocess command for each branch."""

    def setup_method(self):
        self._models = []

    def _make_model(self, **kwargs):
        defaults = {
            "model_id": "test/model",
            "tp_size": 1,
            "concurrency": 16,
            "max_frames": 32,
            "model_type": "qwen",
        }
        defaults.update(kwargs)
        model = mdl.VLLMModel(**defaults)
        model._port = 12345
        import tempfile

        model._log = tempfile.NamedTemporaryFile(
            mode="w", prefix="test_vllm_", suffix=".log", delete=True
        )
        self._models.append(model)
        return model

    def test_uses_sys_executable(self):
        """vllm subprocess always invokes sys.executable directly (single env)."""
        model = self._make_model()

        with unittest.mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = unittest.mock.MagicMock()
            model._start_server()

        cmd = mock_popen.call_args[0][0]
        assert "-m" in cmd
        assert "vllm.entrypoints.openai.api_server" in cmd
        # Should NOT be wrapped in ["bash", "-c", ...]
        assert cmd[0] != "bash"

    def test_qwen_includes_mm_processor_kwargs(self):
        """model_type='qwen' -> --mm-processor-kwargs is present."""
        model = self._make_model(model_type="qwen")

        with unittest.mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = unittest.mock.MagicMock()
            model._start_server()

        cmd = mock_popen.call_args[0][0]
        assert "--mm-processor-kwargs" in cmd

    def test_non_qwen_excludes_mm_processor_kwargs(self):
        """model_type='llama4' -> --mm-processor-kwargs is absent."""
        model = self._make_model(model_type="llama4")

        with unittest.mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = unittest.mock.MagicMock()
            model._start_server()

        cmd = mock_popen.call_args[0][0]
        assert "--mm-processor-kwargs" not in cmd

    def teardown_method(self):
        """Close temp log files created by _make_model."""
        for model in self._models:
            if hasattr(model, "_log"):
                try:
                    model._log.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Tests: VLLMModel _wait_for_health and _verify_served_model
# ---------------------------------------------------------------------------


class TestVLLMHealthCheck:
    """Verify health check logic without launching a real vLLM server."""

    def setup_method(self):
        self._models = []

    def _make_model(self, model_id="test/model"):
        model = mdl.VLLMModel(model_id, tp_size=1)
        model._port = 12345
        import tempfile

        model._log = tempfile.NamedTemporaryFile(
            mode="w", prefix="test_vllm_", suffix=".log", delete=True
        )
        self._models.append(model)
        return model

    def teardown_method(self):
        """Close temp log files created by _make_model."""
        for model in self._models:
            if hasattr(model, "_log"):
                try:
                    model._log.close()
                except Exception:
                    pass

    def test_process_crash_raises_runtime_error(self):
        """Server exits early -> RuntimeError with exit code."""
        model = self._make_model()
        mock_proc = unittest.mock.MagicMock()
        mock_proc.poll.return_value = 1  # process exited with code 1
        mock_proc.returncode = 1
        mock_proc.pid = 99999
        model._proc = mock_proc

        with (
            unittest.mock.patch("os.getpgid", return_value=99999),
            unittest.mock.patch("os.killpg"),
            pytest.raises(RuntimeError, match="exited with code 1"),
        ):
            model._wait_for_health()

    def test_successful_startup(self):
        """Health check 200 + correct model -> returns normally."""
        model = self._make_model()
        mock_proc = unittest.mock.MagicMock()
        mock_proc.poll.return_value = None  # process is running
        mock_proc.pid = 99999
        model._proc = mock_proc

        # Mock both health endpoint and /v1/models endpoint
        health_response = unittest.mock.MagicMock()
        health_response.status = 200
        health_response.__enter__ = lambda s: s
        health_response.__exit__ = lambda s, *a: None

        models_response = unittest.mock.MagicMock()
        models_response.status = 200
        models_data = json.dumps({"data": [{"id": "test/model"}]}).encode()
        models_response.read.return_value = models_data
        models_response.__enter__ = lambda s: s
        models_response.__exit__ = lambda s, *a: None

        def urlopen_side_effect(req, **kwargs):
            if "/health" in req.full_url:
                return health_response
            elif "/v1/models" in req.full_url:
                return models_response
            raise ValueError(f"Unexpected URL: {req.full_url}")

        with unittest.mock.patch(
            "urllib.request.urlopen", side_effect=urlopen_side_effect
        ) as mock_urlopen:
            model._wait_for_health()
            assert mock_urlopen.call_count == 2

    def test_port_collision_raises_runtime_error(self):
        """Health 200 but /v1/models serves a different model -> RuntimeError."""
        model = self._make_model(model_id="expected/model")
        mock_proc = unittest.mock.MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        model._proc = mock_proc

        health_response = unittest.mock.MagicMock()
        health_response.status = 200
        health_response.__enter__ = lambda s: s
        health_response.__exit__ = lambda s, *a: None

        # /v1/models returns a different model
        models_response = unittest.mock.MagicMock()
        models_data = json.dumps({"data": [{"id": "wrong/model"}]}).encode()
        models_response.read.return_value = models_data
        models_response.__enter__ = lambda s: s
        models_response.__exit__ = lambda s, *a: None

        def urlopen_side_effect(req, **kwargs):
            if "/health" in req.full_url:
                return health_response
            elif "/v1/models" in req.full_url:
                return models_response
            raise ValueError(f"Unexpected URL: {req.full_url}")

        with unittest.mock.patch(
            "urllib.request.urlopen", side_effect=urlopen_side_effect
        ):
            with pytest.raises(RuntimeError, match="port collision"):
                model._wait_for_health()


class TestFlatImagesOrdering:
    """Verify that flatten_batch_images() produces correctly ordered flat list."""

    def test_batch_flat_images_positional_ordering(self):
        """Batch of 2 with different image counts produces correctly ordered flat list."""
        from unittest.mock import MagicMock

        # Create mock images with identifiable markers
        img_a1 = MagicMock(name="img_a1")
        img_a2 = MagicMock(name="img_a2")
        img_a3 = MagicMock(name="img_a3")
        img_b1 = MagicMock(name="img_b1")
        img_b2 = MagicMock(name="img_b2")

        batch_frames = [[img_a1, img_a2, img_a3], [img_b1, img_b2]]

        # Call production helper instead of replicating logic inline
        flat_images = mdl.flatten_batch_images(batch_frames)
        assert flat_images == [img_a1, img_a2, img_a3, img_b1, img_b2], (
            "flat_images must concatenate per-conversation images in batch order"
        )

    def test_batch_empty_images_in_middle(self):
        """Conversation with no images doesn't shift other conversations' image positions."""
        from unittest.mock import MagicMock

        img_a1 = MagicMock(name="img_a1")
        img_c1 = MagicMock(name="img_c1")
        img_c2 = MagicMock(name="img_c2")
        batch_frames = [
            [img_a1],
            [],
            [img_c1, img_c2],
        ]
        flat_images = mdl.flatten_batch_images(batch_frames)
        assert flat_images == [img_a1, img_c1, img_c2]


class TestSetupGpus:
    """Tests for model.setup_gpus() minimum GPU enforcement."""

    def test_llama4_below_minimum_raises(self):
        """Llama4 with fewer than 8 GPUs should fail-fast before model load."""
        with unittest.mock.patch.object(mdl, "detect_gpu_count", return_value=2):
            with pytest.raises(RuntimeError, match="llama4 requires at least 8 GPUs"):
                mdl.setup_gpus(num_gpus=None, model_type="llama4")

    def test_llama4_explicit_1gpu_raises(self):
        with unittest.mock.patch.object(mdl, "detect_gpu_count", return_value=8):
            with pytest.raises(RuntimeError, match="llama4 requires at least 8 GPUs"):
                mdl.setup_gpus(num_gpus=1, model_type="llama4")

    def test_llama4_exact_minimum_ok(self):
        with unittest.mock.patch.object(mdl, "detect_gpu_count", return_value=8):
            result = mdl.setup_gpus(num_gpus=8, model_type="llama4")
            assert result == 8

    def test_qwen_1gpu_ok(self):
        with unittest.mock.patch.object(mdl, "detect_gpu_count", return_value=1):
            result = mdl.setup_gpus(num_gpus=None, model_type="qwen")
            assert result == 1

    def test_exceeds_available_raises(self):
        with unittest.mock.patch.object(mdl, "detect_gpu_count", return_value=4):
            with pytest.raises(RuntimeError, match="only 4 available"):
                mdl.setup_gpus(num_gpus=8, model_type="qwen")

    def test_unknown_model_type_defaults_to_1(self):
        with unittest.mock.patch.object(mdl, "detect_gpu_count", return_value=1):
            result = mdl.setup_gpus(num_gpus=None, model_type="custom_model")
            assert result == 1

    def test_num_gpus_less_than_available_sets_cuda_visible_devices(self, monkeypatch):
        """When num_gpus < available, CUDA_VISIBLE_DEVICES should be set."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with (
            unittest.mock.patch.object(mdl, "detect_gpu_count", return_value=4),
            unittest.mock.patch.dict(os.environ, {}, clear=False),
        ):
            result = mdl.setup_gpus(num_gpus=2, model_type="qwen")
            assert result == 2
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_existing_cuda_visible_devices_sliced(self, monkeypatch):
        """When CUDA_VISIBLE_DEVICES is pre-set, slice from existing IDs."""
        with (
            unittest.mock.patch.object(mdl, "detect_gpu_count", return_value=4),
            unittest.mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,5,7"}, clear=False
            ),
        ):
            result = mdl.setup_gpus(num_gpus=2, model_type="qwen")
            assert result == 2
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


class TestGenerateBatchSlicing:
    """Tests for generate_batch decode slicing correctness.

    Exercises the actual Llama4ScoutModel.generate_batch() method with
    mocked processor and model to verify decode slicing uses
    input_ids.shape[1] (padded prompt length).
    """

    def test_generate_batch_correct_decode_slicing(self):
        """Mock model internals and call generate_batch() end-to-end."""
        from unittest.mock import MagicMock, patch

        import torch

        with patch.object(mdl.Llama4ScoutModel, "__init__", return_value=None):
            model = mdl.Llama4ScoutModel.__new__(mdl.Llama4ScoutModel)

        model.processor = MagicMock()
        model.model = MagicMock()
        model.model.device = "cpu"
        model.processor.apply_chat_template = MagicMock(return_value="text")

        pad_id = 0
        input_ids = torch.tensor(
            [
                [pad_id, pad_id, 101, 102, 103],
                [201, 202, 203, 204, 205],
            ]
        )
        model.processor.return_value = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
        }

        output_ids = torch.tensor(
            [
                [pad_id, pad_id, 101, 102, 103, 301, 302, 303],
                [201, 202, 203, 204, 205, 401, 402, 403],
            ]
        )
        model.model.generate.return_value = output_ids

        decode_map = {
            (301, 302, 303): " answer_one ",
            (401, 402, 403): " answer_two ",
        }
        model.processor.decode = MagicMock(
            side_effect=lambda t, **kw: decode_map[tuple(t.tolist())]
        )

        results = model.generate_batch(
            [[], []],
            [
                [{"role": "user", "content": "q1"}],
                [{"role": "user", "content": "q2"}],
            ],
        )

        assert results == ["answer_one", "answer_two"]
        assert model.model.generate.call_count == 1
        assert model.processor.decode.call_count == 2

    def test_qwen_generate_batch_correct_decode_slicing(self):
        """Mock Qwen model internals and call generate_batch() end-to-end.

        Verifies that output_ids[i][input_ids.shape[1]:] correctly extracts
        only generated tokens, matching the Llama4 test above.
        """
        import sys
        import types
        from unittest.mock import MagicMock, patch

        import torch

        # Mock qwen_vl_utils module before it gets imported inside generate_batch
        mock_qwen_utils = types.ModuleType("qwen_vl_utils")
        mock_qwen_utils.process_vision_info = MagicMock(return_value=([], []))

        with patch.dict(sys.modules, {"qwen_vl_utils": mock_qwen_utils}):
            with patch.object(mdl.Qwen2VLModel, "__init__", return_value=None):
                model = mdl.Qwen2VLModel.__new__(mdl.Qwen2VLModel)

            model.processor = MagicMock()
            model.model = MagicMock()
            model.model.device = "cpu"
            model.processor.apply_chat_template = MagicMock(return_value="text")

            pad_id = 0
            input_ids = torch.tensor(
                [
                    [pad_id, pad_id, 101, 102, 103],
                    [201, 202, 203, 204, 205],
                ]
            )

            # Mock processor call to return dict-like object with .to() method
            proc_result = MagicMock()
            proc_result.to = MagicMock(
                return_value={
                    "input_ids": input_ids,
                    "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
                }
            )
            model.processor.return_value = proc_result

            output_ids = torch.tensor(
                [
                    [pad_id, pad_id, 101, 102, 103, 501, 502, 503],
                    [201, 202, 203, 204, 205, 601, 602, 603],
                ]
            )
            model.model.generate.return_value = output_ids

            decode_map = {
                (501, 502, 503): " qwen_answer_one ",
                (601, 602, 603): " qwen_answer_two ",
            }
            model.processor.decode = MagicMock(
                side_effect=lambda t, **kw: decode_map[tuple(t.tolist())]
            )

            results = model.generate_batch(
                [[], []],
                [
                    [{"role": "user", "content": "q1"}],
                    [{"role": "user", "content": "q2"}],
                ],
            )

            assert results == ["qwen_answer_one", "qwen_answer_two"]
            assert model.model.generate.call_count == 1
            assert model.processor.decode.call_count == 2
