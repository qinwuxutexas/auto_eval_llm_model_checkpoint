from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

from .base import JudgeClient


class MockJudgeClient(JudgeClient):
    """
    Deterministic dry-run judge for pipeline / CI smoke tests.

    Scores depend only on (question, response[, stage]) via SHA-256 so
    identical inputs always reproduce identical outputs.
    """

    @staticmethod
    def _u(text: str) -> float:
        h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return (h % 10000) / 10000.0

    def pointwise(
        self,
        image_path: str,
        question: str,
        response: str,
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        u = self._u(question + "|" + response)
        score = 0.3 + 0.65 * u
        return {
            "score": round(score, 4),
            "factuality": round(score, 4),
            "relevance": round(min(1.0, score + 0.03), 4),
            "usefulness": round(score, 4),
            "grounding": round(max(0.0, score - 0.02), 4),
            "hallucination_penalty": round(1.0 - score, 4),
            "rationale": "MOCK",
        }

    def listwise(
        self,
        image_path: str,
        question: str,
        candidates: Dict[str, str],
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        scores = {
            cid: self._u(question + "|" + txt + "|listwise")
            for cid, txt in candidates.items()
        }
        ranking = sorted(scores, key=scores.get, reverse=True)
        return {
            "ranking": ranking,
            "rationale": {k: "MOCK" for k in ranking},
        }

    def pairwise(
        self,
        image_path: str,
        question: str,
        response_a: str,
        response_b: str,
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        a = self._u(question + "|" + response_a + "|pairwise")
        b = self._u(question + "|" + response_b + "|pairwise")
        if abs(a - b) < 0.02:
            winner = "TIE"
        else:
            winner = "A" if a > b else "B"
        return {
            "winner": winner,
            "confidence": round(abs(a - b), 4),
            "rationale": "MOCK",
        }

    def evaluability(
        self,
        image_path: str,
        question: str,
    ) -> Dict[str, Any]:
        u = self._u(str(image_path) + "|" + question + "|eval")
        return {
            "visual_evaluability": round(u, 4),
            "required_evidence_readability": round(min(1.0, u + 0.05), 4),
            "answerability": round(max(0.0, u - 0.05), 4),
            "reason": "MOCK",
        }
