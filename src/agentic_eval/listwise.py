
import json
from pathlib import Path

import pandas as pd

from .data import shuffled_candidate_order


def borda_points(ranking):
    """
    Highest-ranked candidate gets n-1 points,
    next gets n-2, ..., last gets 0.
    """
    n = len(ranking)

    return {
        candidate: n - rank_idx - 1
        for rank_idx, candidate in enumerate(ranking)
    }


def run_listwise(
    samples,
    candidates,
    judge,
    cfg,
    cache_path=None,
):

    score_rows = []
    detail_rows = []

    for sample in samples:

        sample_id = sample["sample_id"]

        available = [
            ckpt
            for ckpt in candidates
            if ckpt in sample["responses"]
        ]

        order = shuffled_candidate_order(
            available,
            sample_id,
            stage="listwise",
            seed=cfg.seed,
        )

        # Anonymous labels shown to the judge
        labels = [
            f"C{i+1}"
            for i in range(len(order))
        ]

        label_to_ckpt = dict(
            zip(labels, order)
        )

        candidate_payload = {
            label: sample["responses"][ckpt]
            for label, ckpt in label_to_ckpt.items()
        }

        result = judge.listwise(
            image_path=sample["image_path"],
            question=sample["question"],
            candidates=candidate_payload,
            model_label=cfg.listwise_judge_label,
            temperature=cfg.listwise_judge_temperature,
        )

        ranking_labels = result["ranking"]

        points = borda_points(
            ranking_labels
        )

        for rank_idx, label in enumerate(
            ranking_labels,
            start=1,
        ):

            ckpt = label_to_ckpt[label]

            score_rows.append({
                "sample_id": sample_id,
                "checkpoint": ckpt,
                "rank": rank_idx,
                "borda": points[label],
            })

        detail_rows.append({
            "sample_id": sample_id,
            "label_to_checkpoint":
                json.dumps(label_to_ckpt),
            "ranking_labels":
                json.dumps(ranking_labels),
            "rationale":
                json.dumps(
                    result.get(
                        "rationale",
                        {}
                    )
                ),
        })

    score_df = pd.DataFrame(
        score_rows
    )

    detail_df = pd.DataFrame(
        detail_rows
    )

    if cache_path:

        cache_path = Path(
            cache_path
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        score_df.to_csv(
            cache_path,
            index=False,
        )

        detail_df.to_csv(
            cache_path.with_name(
                cache_path.stem
                + "_detail.csv"
            ),
            index=False,
        )

    return score_df, detail_df


def summarize_listwise(df):

    summary = (
        df
        .groupby("checkpoint")
        .agg(
            mean_borda=("borda", "mean"),
            mean_rank=("rank", "mean"),
            top1_rate=(
                "rank",
                lambda x: (x == 1).mean(),
            ),
            count=("rank", "count"),
        )
        .sort_values(
            ["mean_borda", "top1_rate"],
            ascending=[False, False],
        )
        .reset_index()
    )

    return summary


def select_finalists(
    summary,
    n_finalists=2,
):

    return (
        summary
        .head(n_finalists)
        ["checkpoint"]
        .tolist()
    )
