
from pathlib import Path

import pandas as pd


def run_quality_evaluation(
    samples,
    judge,
    output_path,
):
    """
    Evaluate IMAGE-QUESTION quality once per sample.

    Checkpoint responses are intentionally NOT used.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Resume previous run
    # ------------------------------------------------------------

    if output_path.exists():

        existing = pd.read_csv(output_path)

        completed = set(
            existing.loc[
                existing["status"] == "ok",
                "sample_id",
            ]
        )

        rows = existing.to_dict("records")

        print(
            f"Resuming: {len(completed)} "
            "samples already completed."
        )

    else:

        completed = set()
        rows = []

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    for i, sample in enumerate(samples):

        sample_id = sample["sample_id"]

        if sample_id in completed:
            continue

        print(
            f"[{i + 1}/{len(samples)}] "
            f"{sample_id}"
        )

        try:

            result = judge.evaluability(
                image_path=sample["image_path"],
                question=sample["question"],
            )

            row = {
                "sample_id":
                    sample_id,

                "page_number":
                    sample.get("page_number"),

                "image_path":
                    sample["image_path"],

                "question":
                    sample["question"],

                "visual_evaluability":
                    float(
                        result[
                            "visual_evaluability"
                        ]
                    ),

                "required_evidence_readability":
                    float(
                        result[
                            "required_evidence_readability"
                        ]
                    ),

                "answerability":
                    float(
                        result[
                            "answerability"
                        ]
                    ),

                "reason":
                    result.get(
                        "reason",
                        "",
                    ),

                "status":
                    "ok",
            }

        except Exception as e:

            print(
                f"ERROR {sample_id}: {e}"
            )

            row = {
                "sample_id":
                    sample_id,

                "page_number":
                    sample.get("page_number"),

                "image_path":
                    sample["image_path"],

                "question":
                    sample["question"],

                "visual_evaluability":
                    None,

                "required_evidence_readability":
                    None,

                "answerability":
                    None,

                "reason":
                    str(e),

                "status":
                    "error",
            }

        rows.append(row)

        # Save after every call
        pd.DataFrame(rows).to_csv(
            output_path,
            index=False,
        )

    return pd.DataFrame(rows)


def add_quality_buckets(df):

    df = df.copy()

    def bucket(x):

        if pd.isna(x):
            return "error"

        if x < 0.30:
            return "low"

        if x < 0.70:
            return "medium"

        return "high"

    df["quality_bucket"] = (
        df["visual_evaluability"]
        .apply(bucket)
    )

    return df


def summarize_quality(df):

    valid = df[
        df["status"] == "ok"
    ].copy()

    valid = add_quality_buckets(valid)

    summary = (
        valid
        .groupby("quality_bucket")
        .agg(
            n=("sample_id", "count"),

            mean_evaluability=(
                "visual_evaluability",
                "mean",
            ),

            mean_required_evidence_readability=(
                "required_evidence_readability",
                "mean",
            ),

            mean_answerability=(
                "answerability",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["percent"] = (
        100.0
        * summary["n"]
        / len(valid)
    )

    order = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }

    summary["_order"] = (
        summary["quality_bucket"]
        .map(order)
    )

    summary = (
        summary
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return summary
