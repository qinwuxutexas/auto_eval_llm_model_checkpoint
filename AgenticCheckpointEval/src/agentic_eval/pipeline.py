
from pathlib import Path

from .data import save_json, build_checkpoint_mapping
from .pointwise import (
    run_pointwise,
    summarize_pointwise,
    select_after_pointwise,
)
from .listwise import (
    run_listwise,
    summarize_listwise,
    select_finalists,
)
from .pairwise import (
    run_pairwise,
    pairwise_win_rate,
)
from .stability import (
    bootstrap_top1_consistency,
    bootstrap_score_ci,
    bootstrap_pairwise_probability,
)
from .metrics import (
    checkpoint_percentile_scores,
    beta_gamma_ablation,
    summarize_ablation_top1,
    validation_loss_baseline,
)


def run_agentic_pipeline(
    samples,
    judge,
    cfg,
    out_dir,
    loss_by_checkpoint=None,
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Checkpoint mapping
    # =========================================================

    mapping = build_checkpoint_mapping(
        samples,
        out_path=out_dir / "checkpoint_mapping.json",
        seed=cfg.seed,
    )

    # =========================================================
    # STAGE I: POINTWISE
    # =========================================================

    print("\n========== STAGE I: POINTWISE ==========")

    pointwise_df = run_pointwise(
        samples=samples,
        judge=judge,
        cfg=cfg,
        cache_path=out_dir / "stage1_pointwise.csv",
    )

    pointwise_summary = summarize_pointwise(
        pointwise_df
    )

    pointwise_summary.to_csv(
        out_dir / "stage1_pointwise_summary.csv",
        index=False,
    )

    print(pointwise_summary)

    kept = select_after_pointwise(
        pointwise_summary,
        cfg,
    )

    print("\nKept after pointwise:")
    print(kept)

    save_json(
        {"kept_checkpoints": kept},
        out_dir / "stage1_selection.json",
    )

    # Bootstrap stability
    pointwise_stability = bootstrap_top1_consistency(
        pointwise_df,
        value_col="score",
        higher_is_better=True,
        rounds=cfg.bootstrap_rounds,
        seed=cfg.seed,
    )

    save_json(
        pointwise_stability,
        out_dir / "stage1_stability.json",
    )

    # Score confidence intervals
    pointwise_ci = bootstrap_score_ci(
        pointwise_df,
        value_col="score",
        rounds=cfg.bootstrap_rounds,
        seed=cfg.seed,
    )

    save_json(
        pointwise_ci,
        out_dir / "stage1_score_ci.json",
    )

    # =========================================================
    # Percentile stability score
    # =========================================================

    percentile_scores = checkpoint_percentile_scores(
        pointwise_df,
        value_col="score",
        beta=cfg.beta,
        gamma=cfg.gamma,
    )

    percentile_scores.to_csv(
        out_dir / "percentile_scores.csv",
        index=False,
    )

    # beta / gamma ablation
    ablation_df = beta_gamma_ablation(
        pointwise_df,
        value_col="score",
        beta_grid=cfg.beta_grid,
        gamma_grid=cfg.gamma_grid,
    )

    ablation_df.to_csv(
        out_dir / "beta_gamma_ablation.csv",
        index=False,
    )

    ablation_summary = summarize_ablation_top1(
        ablation_df
    )

    ablation_summary.to_csv(
        out_dir / "beta_gamma_ablation_summary.csv",
        index=False,
    )

    # =========================================================
    # STAGE II: LISTWISE
    # =========================================================

    print("\n========== STAGE II: LISTWISE ==========")

    listwise_df, listwise_detail = run_listwise(
        samples=samples,
        candidates=kept,
        judge=judge,
        cfg=cfg,
        cache_path=out_dir / "stage2_listwise.csv",
    )

    listwise_summary = summarize_listwise(
        listwise_df
    )

    listwise_summary.to_csv(
        out_dir / "stage2_listwise_summary.csv",
        index=False,
    )

    print(listwise_summary)

    listwise_stability = bootstrap_top1_consistency(
        listwise_df,
        value_col="borda",
        higher_is_better=True,
        rounds=cfg.bootstrap_rounds,
        seed=cfg.seed + 1,
    )

    save_json(
        listwise_stability,
        out_dir / "stage2_stability.json",
    )

    finalists = select_finalists(
        listwise_summary,
        n_finalists=cfg.listwise_finalists,
    )

    print("\nFinalists:")
    print(finalists)

    save_json(
        {"finalists": finalists},
        out_dir / "stage2_selection.json",
    )

    # =========================================================
    # STAGE III: PAIRWISE
    # =========================================================

    print("\n========== STAGE III: PAIRWISE ==========")

    pairwise_summary = {}
    pairwise_bootstrap = {}

    if len(finalists) >= 2:

        ckpt_a = finalists[0]
        ckpt_b = finalists[1]

        pairwise_df = run_pairwise(
            samples=samples,
            ckpt_a=ckpt_a,
            ckpt_b=ckpt_b,
            judge=judge,
            cfg=cfg,
            cache_path=out_dir / "stage3_pairwise.csv",
        )

        pairwise_summary = pairwise_win_rate(
            pairwise_df,
            ckpt_a,
            ckpt_b,
        )

        pairwise_bootstrap = bootstrap_pairwise_probability(
            pairwise_df,
            ckpt_a,
            ckpt_b,
            rounds=cfg.bootstrap_rounds,
            seed=cfg.seed + 2,
        )

        save_json(
            pairwise_summary,
            out_dir / "stage3_pairwise_summary.json",
        )

        save_json(
            pairwise_bootstrap,
            out_dir / "stage3_pairwise_bootstrap.json",
        )

        print(pairwise_summary)
        print(pairwise_bootstrap)

    # =========================================================
    # EXTERNAL BASELINE: VALIDATION LOSS
    # =========================================================

    loss_baseline = {}

    if loss_by_checkpoint:

        print("\n========== VALIDATION LOSS BASELINE ==========")

        loss_baseline = validation_loss_baseline(
            loss_by_checkpoint
        )

        save_json(
            loss_baseline,
            out_dir / "validation_loss_baseline.json",
        )

        print(loss_baseline)

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    final_summary = {
        "n_samples": len(samples),
        "n_initial_checkpoints": len(mapping),

        "pointwise_top1":
            pointwise_stability["reference_top1"],

        "pointwise_top1_consistency":
            pointwise_stability["top1_consistency"],

        "pointwise_flip_rate":
            pointwise_stability["flip_rate"],

        "kept_after_pointwise":
            kept,

        "listwise_top1":
            listwise_stability["reference_top1"],

        "listwise_top1_consistency":
            listwise_stability["top1_consistency"],

        "listwise_flip_rate":
            listwise_stability["flip_rate"],

        "finalists":
            finalists,

        "pairwise_summary":
            pairwise_summary,

        "pairwise_bootstrap":
            pairwise_bootstrap,

        "loss_baseline":
            loss_baseline,
    }

    save_json(
        final_summary,
        out_dir / "rebuttal_summary.json",
    )

    print("\n========== FINAL SUMMARY ==========")

    for key, value in final_summary.items():
        print(f"{key}: {value}")

    return final_summary
