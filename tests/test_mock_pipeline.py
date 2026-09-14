from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@pytest.fixture(scope="module")
def samples():
    import sys

    sys.path.insert(0, str(SRC))
    from agentic_eval.data import read_jsonl

    path = ROOT / "data" / "samples" / "mock_mini.jsonl"
    return read_jsonl(path)


def test_mock_pipeline_deterministic(samples, tmp_path):
    import sys

    sys.path.insert(0, str(SRC))
    from agentic_eval.config import EvalConfig
    from agentic_eval.judges import MockJudgeClient
    from agentic_eval.pipeline import run_agentic_pipeline

    cfg = EvalConfig(seed=42, bootstrap_rounds=50)
    judge = MockJudgeClient()

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    sa = run_agentic_pipeline(samples, judge, cfg, out_a)
    sb = run_agentic_pipeline(samples, judge, cfg, out_b)

    assert sa == sb
    assert (out_a / "config.json").exists()
    assert (out_a / "stage1_pointwise.csv").read_bytes() == (
        out_b / "stage1_pointwise.csv"
    ).read_bytes()
    assert json.loads((out_a / "rebuttal_summary.json").read_text())["n_samples"] == 2
