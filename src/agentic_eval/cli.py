from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import EvalConfig, config_from_mapping
from .data import read_jsonl, validate_dataset
from .judges import MockJudgeClient, OpenAIJudgeClient
from .pipeline import run_agentic_pipeline


def _load_config(path: Optional[str], overrides: Dict[str, Any]) -> EvalConfig:
    cfg = EvalConfig()
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = config_from_mapping(raw)
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _build_json_from_pdf(
    pdf_path: str,
    image_folder: str,
    output_jsonl: str,
    image_glob: str = "*",
):
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF (fitz) is required for build-json: pip install pymupdf"
        ) from exc

    from .data import write_jsonl

    pdf = fitz.open(pdf_path)
    images = sorted(Path(image_folder).glob(image_glob))
    if not images:
        raise FileNotFoundError(f"No images matching {image_glob} in {image_folder}")

    rows = []
    n = min(len(pdf), len(images))
    for i in range(n):
        page = pdf[i]
        text = page.get_text("text").strip()
        rows.append(
            {
                "sample_id": f"sample_{i+1:04d}",
                "page_number": i + 1,
                "question": text,
                "image_path": str(images[i].as_posix()),
                "responses": {},
            }
        )
    write_jsonl(rows, output_jsonl)
    return rows


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agentic_eval",
        description="Agentic multimodal LLM checkpoint evaluation",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser(
        "build-json",
        help="Build starter JSONL from PDF pages + aligned image folder",
    )
    p_build.add_argument("--pdf", required=True)
    p_build.add_argument("--image-folder", required=True)
    p_build.add_argument("--image-glob", default="*")
    p_build.add_argument("--out", required=True)

    p_eval = sub.add_parser("evaluate", help="Run Stage I→II→III pipeline")
    p_eval.add_argument("--data", required=True, help="Input JSONL samples")
    p_eval.add_argument("--out-dir", required=True)
    p_eval.add_argument("--config", default=None, help="YAML config path")
    p_eval.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic MockJudgeClient (no API)",
    )
    p_eval.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAIJudgeClient (requires OPENAI_API_KEY)",
    )
    p_eval.add_argument("--model", default="gpt-4o")
    p_eval.add_argument("--image-root", default=".")
    p_eval.add_argument("--seed", type=int, default=None)
    p_eval.add_argument("--bootstrap-rounds", type=int, default=None)
    p_eval.add_argument("--beta", type=float, default=None)
    p_eval.add_argument("--gamma", type=float, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "build-json":
        rows = _build_json_from_pdf(
            pdf_path=args.pdf,
            image_folder=args.image_folder,
            output_jsonl=args.out,
            image_glob=args.image_glob,
        )
        print(f"Wrote {len(rows)} samples to {args.out}")
        return

    if args.cmd == "evaluate":
        samples = read_jsonl(args.data)
        validate_dataset(samples)

        cfg = _load_config(
            args.config,
            {
                "seed": args.seed,
                "bootstrap_rounds": args.bootstrap_rounds,
                "beta": args.beta,
                "gamma": args.gamma,
            },
        )

        if args.mock:
            judge = MockJudgeClient()
        elif args.openai:
            judge = OpenAIJudgeClient(
                model=args.model,
                image_root=args.image_root,
            )
        else:
            raise SystemExit(
                "Select a judge: pass --mock (smoke test) or --openai "
                "(OPENAI_API_KEY). For Gemini/Claude, subclass JudgeClient."
            )

        summary = run_agentic_pipeline(samples, judge, cfg, args.out_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
