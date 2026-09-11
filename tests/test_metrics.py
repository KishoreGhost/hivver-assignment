"""
tests/test_metrics.py
---------------------
Tests for metric calculations (eval/metrics.py).

Feeds known inputs, asserts known outputs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from eval.metrics import (
    intent_metrics,
    escalation_metrics,
    _lcs_len,
    bleu_rouge_scores,
)


class TestIntentMetrics:
    def test_perfect_accuracy(self):
        y = ["a", "b", "c", "a"]
        r = intent_metrics(y, y)
        assert r["accuracy"]  == 1.0
        assert r["macro_f1"]  == 1.0

    def test_all_wrong(self):
        y_true = ["a", "a", "a"]
        y_pred = ["b", "b", "b"]
        r = intent_metrics(y_true, y_pred)
        assert r["accuracy"] == 0.0

    def test_partial_accuracy(self):
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "b", "b", "a"]
        r = intent_metrics(y_true, y_pred)
        assert r["accuracy"] == 0.5

    def test_labels_in_confusion_matrix(self):
        y_true = ["cat", "dog", "cat"]
        y_pred = ["cat", "cat", "dog"]
        r = intent_metrics(y_true, y_pred)
        assert "cat" in r["labels"]
        assert "dog" in r["labels"]
        assert len(r["confusion_matrix"]) == 2

    def test_single_class(self):
        y = ["other"] * 5
        r = intent_metrics(y, y)
        assert r["accuracy"] == 1.0


class TestEscalationMetrics:
    def test_perfect_detection(self):
        y_true = [True, False, True, False]
        y_pred = [True, False, True, False]
        r = escalation_metrics(y_true, y_pred)
        assert r["precision"] == 1.0
        assert r["recall"]    == 1.0
        assert r["f1"]        == 1.0

    def test_all_false_negatives(self):
        """All escalations missed → recall = 0."""
        y_true = [True, True, True]
        y_pred = [False, False, False]
        r = escalation_metrics(y_true, y_pred)
        assert r["recall"]    == 0.0
        assert r["precision"] == 0.0

    def test_false_negative_weight(self):
        """Cost-weighted F1 should be lower than standard F1 when FNs are present."""
        # 1 TP, 1 FN (miss), 0 FP
        y_true = [True, True, False]
        y_pred = [True, False, False]
        r = escalation_metrics(y_true, y_pred, fn_weight=3.0)
        assert r["f1"] > 0
        # Cost-weighted F1 penalises FNs more → should be <= standard recall-weighted
        fw_key = "f1_weighted_fn3.0x"
        assert fw_key in r
        # With high FN weight and a missed case, weighted F1 <= standard F1
        assert r[fw_key] <= r["f1"] + 1e-9

    def test_no_positives_returns_error(self):
        y_true = [False, False]
        y_pred = [False, False]
        r = escalation_metrics(y_true, y_pred)
        assert "error" in r


class TestLCS:
    def test_identical(self):
        assert _lcs_len(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_empty(self):
        assert _lcs_len([], ["a", "b"]) == 0

    def test_partial(self):
        assert _lcs_len(["a", "b", "c"], ["a", "c"]) == 2


class TestBLEUROUGE:
    def test_identical_returns_high_scores(self):
        hyps = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        r = bleu_rouge_scores(hyps, refs)
        if r["rouge_l"] is not None:
            assert r["rouge_l"] > 0.9

    def test_completely_different(self):
        hyps = ["xyz abc"]
        refs = ["the cat sat on the mat"]
        r = bleu_rouge_scores(hyps, refs)
        if r["rouge_l"] is not None:
            assert r["rouge_l"] < 0.5

    def test_note_present(self):
        r = bleu_rouge_scores(["test"], ["reference"])
        assert "note" in r
