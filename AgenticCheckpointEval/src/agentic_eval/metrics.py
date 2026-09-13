
import numpy as np
import pandas as pd


def percentile_stability_score(
    values,
    beta=0.50,
    gamma=0.25,
):
    """
    Paper formulation:

    S = P50
        - beta * (P50 - P20)
        + gamma * (P80 - P50)

    beta:
        penalty on lower-tail instability

    gamma:
        reward on upper-tail performance
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    p20, p50, p80 = np.percentile(
        values,
        [20, 50, 80],
    )

    score = (
        p50
        - beta * (p50 - p20)
        + gamma * (p80 - p50)
    )

    return {
        "P20": float(p20),
        "P50": float(p50),
        "P80": float(p80),
        "score": float(score),
    }


def checkpoint_percentile_scores(
    per_sample_df,
    value_col,
    beta=0.50,
    gamma=0.25,
):
    """
    Compute percentile-based score
    for every checkpoint.
    """

    rows = []

    for checkpoint, group in (
        per_sample_df
        .groupby("checkpoint")
    ):

        result = percentile_stability_score(
            group[value_col].values,
            beta=beta,
            gamma=gamma,
        )

        rows.append({
            "checkpoint": checkpoint,
            "beta": beta,
            "gamma": gamma,
            **result,
        })

    df = pd.DataFrame(rows)

    return (
        df
        .sort_values(
            "score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def beta_gamma_ablation(
    per_sample_df,
    value_col,
    beta_grid=None,
    gamma_grid=None,
):
    """
    Sensitivity analysis over beta/gamma.
    """

    if beta_grid is None:
        beta_grid = [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ]

    if gamma_grid is None:
        gamma_grid = [
            0.0,
            0.10,
            0.25,
            0.50,
        ]

    rows = []

    for beta in beta_grid:

        for gamma in gamma_grid:

            scores = checkpoint_percentile_scores(
                per_sample_df,
                value_col=value_col,
                beta=beta,
                gamma=gamma,
            )

            top1 = scores.iloc[0]["checkpoint"]

            for _, row in scores.iterrows():

                rows.append({
                    "beta": beta,
                    "gamma": gamma,
                    "checkpoint":
                        row["checkpoint"],
                    "P20":
                        row["P20"],
                    "P50":
                        row["P50"],
                    "P80":
                        row["P80"],
                    "stability_score":
                        row["score"],
                    "top1":
                        top1,
                })

    return pd.DataFrame(rows)


def summarize_ablation_top1(
    ablation_df,
):
    """
    How often each checkpoint is selected
    across beta/gamma settings.
    """

    combinations = (
        ablation_df[
            ["beta", "gamma", "top1"]
        ]
        .drop_duplicates()
    )

    summary = (
        combinations["top1"]
        .value_counts()
        .rename_axis("checkpoint")
        .reset_index(
            name="n_selected"
        )
    )

    total = len(combinations)

    summary["selection_rate"] = (
        summary["n_selected"]
        / total
    )

    return summary


def validation_loss_baseline(
    loss_by_checkpoint,
):
    """
    Conventional checkpoint selection baseline.

    Lower validation/training loss is better.

    Input example:

    {
        "ckpt_2000": 1.24,
        "ckpt_4000": 1.18,
        "ckpt_8000": 1.10
    }
    """

    if not loss_by_checkpoint:
        return {}

    ranking = sorted(
        loss_by_checkpoint.items(),
        key=lambda item: item[1],
    )

    best_checkpoint, best_loss = (
        ranking[0]
    )

    return {
        "selected_checkpoint":
            best_checkpoint,

        "selected_loss":
            float(best_loss),

        "ranking":
            [
                checkpoint
                for checkpoint, _
                in ranking
            ],

        "losses":
            {
                checkpoint:
                    float(loss)

                for checkpoint, loss
                in ranking
            },
    }


def compare_loss_vs_eval(
    loss_by_checkpoint,
    eval_summary,
    eval_score_col="mean",
):
    """
    Compare conventional loss-selected checkpoint
    with evaluation-selected checkpoint.
    """

    loss_result = validation_loss_baseline(
        loss_by_checkpoint
    )

    if not loss_result:
        return {}

    eval_sorted = (
        eval_summary
        .sort_values(
            eval_score_col,
            ascending=False,
        )
    )

    eval_selected = (
        eval_sorted
        .iloc[0]["checkpoint"]
    )

    return {
        "loss_selected_checkpoint":
            loss_result[
                "selected_checkpoint"
            ],

        "eval_selected_checkpoint":
            eval_selected,

        "same_selection":
            (
                loss_result[
                    "selected_checkpoint"
                ]
                == eval_selected
            ),
    }
