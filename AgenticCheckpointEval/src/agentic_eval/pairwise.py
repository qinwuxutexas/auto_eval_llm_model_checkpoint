
from pathlib import Path

import numpy as np
import pandas as pd

from .data import shuffled_candidate_order


def run_pairwise(
    samples,
    ckpt_a,
    ckpt_b,
    judge,
    cfg,
    cache_path=None,
):

    rows = []

    for sample in samples:

        if (
            ckpt_a not in sample["responses"]
            or ckpt_b not in sample["responses"]
        ):
            continue

        order = shuffled_candidate_order(
            [ckpt_a, ckpt_b],
            sample["sample_id"],
            stage="pairwise",
            seed=cfg.seed,
        )

        shown_a = order[0]
        shown_b = order[1]

        result = judge.pairwise(
            image_path=sample["image_path"],
            question=sample["question"],
            response_a=
                sample["responses"][shown_a],
            response_b=
                sample["responses"][shown_b],
            model_label="GPT-4o reasoning",
            temperature=
                cfg.pairwise_judge_temperature,
        )

        winner_label = result["winner"]

        if winner_label == "A":
            winner_ckpt = shown_a

        elif winner_label == "B":
            winner_ckpt = shown_b

        else:
            winner_ckpt = "TIE"

        rows.append({
            "sample_id":
                sample["sample_id"],

            "shown_A":
                shown_a,

            "shown_B":
                shown_b,

            "winner":
                winner_ckpt,

            "confidence":
                result.get(
                    "confidence"
                ),

            "rationale":
                result.get(
                    "rationale",
                    ""
                ),
        })

    df = pd.DataFrame(
        rows
    )

    if cache_path:

        cache_path = Path(
            cache_path
        )

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            cache_path,
            index=False,
        )

    return df


def pairwise_win_rate(
    df,
    ckpt_a,
    ckpt_b,
):

    if len(df) == 0:

        return {
            "n": 0,
            "a_win_rate": np.nan,
            "b_win_rate": np.nan,
            "tie_rate": np.nan,
        }

    return {
        "n": len(df),

        "a_win_rate":
            float(
                np.mean(
                    df["winner"]
                    == ckpt_a
                )
            ),

        "b_win_rate":
            float(
                np.mean(
                    df["winner"]
                    == ckpt_b
                )
            ),

        "tie_rate":
            float(
                np.mean(
                    df["winner"]
                    == "TIE"
                )
            ),
    }
