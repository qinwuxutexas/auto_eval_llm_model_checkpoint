
from pathlib import Path

import pandas as pd

from .data import shuffled_candidate_order


def run_pointwise(
    samples,
    judge,
    cfg,
    cache_path=None,
):

    rows = []

    for sample in samples:

        sample_id = sample["sample_id"]

        checkpoints = list(
            sample["responses"].keys()
        )

        order = shuffled_candidate_order(
            checkpoints,
            sample_id,
            stage="pointwise",
            seed=cfg.seed,
        )

        for checkpoint in order:

            result = judge.pointwise(
                image_path=sample["image_path"],
                question=sample["question"],
                response=sample["responses"][checkpoint],
                model_label=cfg.pointwise_judge_label,
                temperature=cfg.pointwise_judge_temperature,
            )

            rows.append({
                "sample_id": sample_id,
                "checkpoint": checkpoint,
                "score": float(result["score"]),
                "factuality": result.get("factuality"),
                "relevance": result.get("relevance"),
                "usefulness": result.get("usefulness"),
                "grounding": result.get("grounding"),
                "hallucination_penalty":
                    result.get("hallucination_penalty"),
                "rationale": result.get("rationale", ""),
            })

    df = pd.DataFrame(rows)

    if cache_path:

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            cache_path,
            index=False,
        )

    return df


def summarize_pointwise(df):

    summary = (
        df
        .groupby("checkpoint")["score"]
        .agg([
            "mean",
            "median",
            "std",
            "count",
        ])
        .sort_values(
            "mean",
            ascending=False,
        )
        .reset_index()
    )

    return summary


def select_after_pointwise(
    summary,
    cfg,
):
    """
    Experimental filtering rule.

    IMPORTANT:
    Replace this with the original historical rule
    once recovered from your previous implementation.
    """

    summary = (
        summary
        .sort_values(
            "mean",
            ascending=False,
        )
        .copy()
    )

    best_score = float(
        summary.iloc[0]["mean"]
    )

    margin = getattr(cfg, "pointwise_keep_margin", 0.10)

    kept = summary[
        summary["mean"]
        >= best_score - margin
    ]["checkpoint"].tolist()

    if len(kept) < cfg.pointwise_min_candidates:

        kept = (
            summary
            .head(
                min(
                    cfg.pointwise_min_candidates,
                    len(summary),
                )
            )["checkpoint"]
            .tolist()
        )

    if len(kept) > cfg.pointwise_max_candidates:

        kept = (
            summary
            .head(
                cfg.pointwise_max_candidates
            )["checkpoint"]
            .tolist()
        )

    return kept
