
import numpy as np
import pandas as pd


def bootstrap_top1_consistency(
    per_sample_df,
    value_col,
    higher_is_better=True,
    rounds=1000,
    seed=42,
):
    """
    Bootstrap samples with replacement.

    Reference top-1:
        checkpoint ranked first using the full available subset.

    Top-1 consistency:
        fraction of bootstrap runs selecting the same top checkpoint.
    """

    rng = np.random.default_rng(seed)

    sample_ids = (
        per_sample_df["sample_id"]
        .unique()
    )

    # Full-subset reference ranking
    reference_scores = (
        per_sample_df
        .groupby("checkpoint")[value_col]
        .mean()
    )

    if higher_is_better:
        reference_scores = reference_scores.sort_values(
            ascending=False
        )
    else:
        reference_scores = reference_scores.sort_values(
            ascending=True
        )

    reference_top1 = reference_scores.index[0]

    winners = []

    for _ in range(rounds):

        sampled_ids = rng.choice(
            sample_ids,
            size=len(sample_ids),
            replace=True,
        )

        # Important:
        # preserve bootstrap multiplicity
        bootstrap_parts = []

        for bootstrap_idx, sample_id in enumerate(
            sampled_ids
        ):

            part = per_sample_df[
                per_sample_df["sample_id"]
                == sample_id
            ].copy()

            part["_bootstrap_idx"] = bootstrap_idx

            bootstrap_parts.append(part)

        bootstrap_df = pd.concat(
            bootstrap_parts,
            ignore_index=True,
        )

        scores = (
            bootstrap_df
            .groupby("checkpoint")[value_col]
            .mean()
        )

        if higher_is_better:
            winner = scores.idxmax()
        else:
            winner = scores.idxmin()

        winners.append(winner)

    winners_array = np.asarray(winners)

    top1_consistency = float(
        np.mean(
            winners_array
            == reference_top1
        )
    )

    winner_frequency = (
        pd.Series(winners)
        .value_counts(normalize=True)
        .to_dict()
    )

    return {
        "reference_top1":
            reference_top1,

        "top1_consistency":
            top1_consistency,

        "flip_rate":
            1.0 - top1_consistency,

        "winner_frequency":
            winner_frequency,

        "rounds":
            rounds,
    }


def bootstrap_score_ci(
    per_sample_df,
    value_col,
    rounds=1000,
    seed=42,
):
    """
    95% bootstrap confidence interval
    for each checkpoint's mean score.
    """

    rng = np.random.default_rng(seed)

    output = {}

    for checkpoint, group in (
        per_sample_df
        .groupby("checkpoint")
    ):

        values = (
            group[value_col]
            .astype(float)
            .to_numpy()
        )

        n = len(values)

        bootstrap_means = []

        for _ in range(rounds):

            sampled = rng.choice(
                values,
                size=n,
                replace=True,
            )

            bootstrap_means.append(
                np.mean(sampled)
            )

        bootstrap_means = np.asarray(
            bootstrap_means
        )

        output[checkpoint] = {
            "mean":
                float(np.mean(values)),

            "ci_low":
                float(
                    np.quantile(
                        bootstrap_means,
                        0.025,
                    )
                ),

            "ci_high":
                float(
                    np.quantile(
                        bootstrap_means,
                        0.975,
                    )
                ),
        }

    return output


def bootstrap_pairwise_probability(
    pairwise_df,
    ckpt_a,
    ckpt_b,
    rounds=1000,
    seed=42,
):
    """
    Bootstrap estimate of pairwise preference.

    A win -> 1
    B win -> 0
    tie   -> 0.5

    Returns:
        mean preference score for A
        P(A > B)
        95% bootstrap CI
    """

    rng = np.random.default_rng(seed)

    values = []

    for winner in pairwise_df["winner"]:

        if winner == ckpt_a:
            values.append(1.0)

        elif winner == ckpt_b:
            values.append(0.0)

        else:
            values.append(0.5)

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    bootstrap_scores = []

    for _ in range(rounds):

        sampled = rng.choice(
            values,
            size=n,
            replace=True,
        )

        bootstrap_scores.append(
            np.mean(sampled)
        )

    bootstrap_scores = np.asarray(
        bootstrap_scores
    )

    return {
        "mean_preference_A":
            float(np.mean(values)),

        "P_A_gt_B":
            float(
                np.mean(
                    bootstrap_scores > 0.5
                )
            ),

        "ci_low":
            float(
                np.quantile(
                    bootstrap_scores,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    bootstrap_scores,
                    0.975,
                )
            ),

        "rounds":
            rounds,
    }


def ranking_from_scores(
    df,
    value_col,
    higher_is_better=True,
):
    """
    Return checkpoint ranking.
    """

    scores = (
        df
        .groupby("checkpoint")[value_col]
        .mean()
    )

    scores = scores.sort_values(
        ascending=not higher_is_better
    )

    return scores.index.tolist()


def pairwise_ranking_agreement(
    ranking_a,
    ranking_b,
):
    """
    Fraction of checkpoint pairs
    whose relative ordering agrees.

    Range:
        0 = complete disagreement
        1 = complete agreement
    """

    common = [
        ckpt
        for ckpt in ranking_a
        if ckpt in ranking_b
    ]

    pos_a = {
        ckpt: idx
        for idx, ckpt
        in enumerate(ranking_a)
    }

    pos_b = {
        ckpt: idx
        for idx, ckpt
        in enumerate(ranking_b)
    }

    total = 0
    agreement = 0

    for i in range(len(common)):

        for j in range(
            i + 1,
            len(common),
        ):

            ckpt_i = common[i]
            ckpt_j = common[j]

            order_a = (
                pos_a[ckpt_i]
                < pos_a[ckpt_j]
            )

            order_b = (
                pos_b[ckpt_i]
                < pos_b[ckpt_j]
            )

            total += 1

            if order_a == order_b:
                agreement += 1

    if total == 0:
        return float("nan")

    return agreement / total
