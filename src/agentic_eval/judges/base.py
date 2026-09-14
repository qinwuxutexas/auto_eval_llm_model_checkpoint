from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class JudgeClient(ABC):
    """
    Vendor-agnostic JSON interface used by all pipeline stages.

    Implement pointwise / listwise / pairwise against your API of choice.
    """

    @abstractmethod
    def pointwise(
        self,
        image_path: str,
        question: str,
        response: str,
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def listwise(
        self,
        image_path: str,
        question: str,
        candidates: Dict[str, str],
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def pairwise(
        self,
        image_path: str,
        question: str,
        response_a: str,
        response_b: str,
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def evaluability(
        self,
        image_path: str,
        question: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "evaluability() is optional; implement for quality filtering."
        )
