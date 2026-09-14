from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class EvalConfig:
    """Protocol knobs for the 3-stage agentic checkpoint evaluation."""

    seed: int = 42

    # Public reproducibility subset size (paper figures use 100 images).
    public_subset_size: int = 100

    # Original paper subsample settings (recover exact values before claims).
    original_subsample_size: int = 800
    original_resampling_rounds: Optional[int] = None
    original_with_replacement: Optional[bool] = None

    bootstrap_rounds: int = 1000
    bootstrap_with_replacement: bool = True

    pointwise_keep_margin: float = 0.10
    pointwise_min_candidates: int = 4
    pointwise_max_candidates: int = 6

    listwise_finalists: int = 2
    pairwise_confidence_threshold: float = 0.70

    # S = P50 - beta*(P50-P20) + gamma*(P80-P50)
    beta: float = 0.50
    gamma: float = 0.25
    beta_grid: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    gamma_grid: Tuple[float, ...] = (0.0, 0.10, 0.25, 0.50)

    randomize_candidate_order: bool = True

    # Labels only — map to real API model IDs in judge clients.
    pointwise_judge_label: str = "Gemini 3 Flash"
    listwise_judge_label: str = "Claude Sonnet 4.1"
    pairwise_judge_label: str = "GPT-4o reasoning"

    pointwise_judge_temperature: Optional[float] = 0.0
    listwise_judge_temperature: Optional[float] = 0.0
    pairwise_judge_temperature: Optional[float] = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def config_from_mapping(mapping: Dict[str, Any]) -> EvalConfig:
    """Build EvalConfig from a plain dict (e.g. YAML/JSON), ignoring unknown keys."""
    known = {f.name for f in EvalConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in mapping.items() if k in known}
    # tuples from YAML lists
    for key in ("beta_grid", "gamma_grid"):
        if key in filtered and isinstance(filtered[key], list):
            filtered[key] = tuple(filtered[key])
    return EvalConfig(**filtered)
