
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EvalConfig:
    seed: int = 42

    # New 100-example reproducibility analysis
    bootstrap_rounds: int = 1000

    # Initial experimental values; ablate later
    beta: float = 0.50
    gamma: float = 0.25

    beta_grid: Tuple[float, ...] = (
        0.0, 0.25, 0.50, 0.75, 1.0
    )

    gamma_grid: Tuple[float, ...] = (
        0.0, 0.10, 0.25, 0.50
    )

    # Recover exact historical values before manuscript claims
    pointwise_judge_temperature: Optional[float] = None
    listwise_judge_temperature: Optional[float] = None
    pairwise_judge_temperature: Optional[float] = None

    pointwise_min_candidates: int = 4
    pointwise_max_candidates: int = 6

    listwise_finalists: int = 2
