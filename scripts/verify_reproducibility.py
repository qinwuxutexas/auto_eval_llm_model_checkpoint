#!/usr/bin/env python3
"""Re-run the mock pipeline twice and assert byte-identical key artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_eval.config import EvalConfig
from agentic_eval.data import read_jsonl
from agentic_eval.judges import MockJudgeClient
from agentic_eval.pipeline import run_agentic_pipeline


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = ROOT / "data" / "samples" / "mock_mini.jsonl"
    if not data.exists():
        raise SystemExit(f"Missing fixture: {data}")

    samples = read_jsonl(data)
    cfg = EvalConfig(seed=42, bootstrap_rounds=200)
    judge = MockJudgeClient()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        run_agentic_pipeline(samples, judge, cfg, out_a)
        run_agentic_pipeline(samples, judge, cfg, out_b)

        keys = [
            "stage1_pointwise.csv",
            "stage2_listwise.csv",
            "stage3_pairwise.csv",
            "rebuttal_summary.json",
            "checkpoint_mapping.json",
            "config.json",
        ]
        for name in keys:
            ha = _file_hash(out_a / name)
            hb = _file_hash(out_b / name)
            if ha != hb:
                raise SystemExit(f"Non-deterministic artifact: {name}")

        summary = json.loads((out_a / "rebuttal_summary.json").read_text(encoding="utf-8"))
        print("Determinism OK")
        print(json.dumps(summary, indent=2))

        # Optionally refresh checked-in smoke results
        dest = ROOT / "results" / "mock_smoke"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out_a, dest)
        print(f"Wrote reproducible smoke results to {dest}")


if __name__ == "__main__":
    main()
