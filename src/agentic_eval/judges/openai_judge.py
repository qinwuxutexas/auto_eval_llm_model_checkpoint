from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ..prompts import (
    EVALUABILITY_RUBRIC,
    LISTWISE_RUBRIC,
    PAIRWISE_RUBRIC,
    POINTWISE_RUBRIC,
)
from .base import JudgeClient


class OpenAIJudgeClient(JudgeClient):
    """
    OpenAI Responses API judge (gpt-4o by default).

    Set OPENAI_API_KEY in the environment. image_root resolves relative paths.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        image_root: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install openai to use OpenAIJudgeClient: pip install openai"
            ) from exc

        import os

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIJudgeClient")

        self.client = OpenAI(api_key=key)
        self.model = model
        self.image_root = Path(image_root) if image_root else Path.cwd()

    def _encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.image_root / path
        if not path.exists():
            raise FileNotFoundError(path)
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
        return json.loads(text)

    def _call(
        self,
        prompt: str,
        image_path: str,
        temperature: Optional[float],
    ) -> Dict[str, Any]:
        image_b64 = self._encode_image(image_path)
        temp = 0.0 if temperature is None else float(temperature)
        result = self.client.responses.create(
            model=self.model,
            temperature=temp,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_b64}",
                        },
                    ],
                }
            ],
        )
        return self._parse_json(result.output_text)

    def evaluability(self, image_path: str, question: str) -> Dict[str, Any]:
        prompt = (
            f"{EVALUABILITY_RUBRIC}\n\nQUESTION:\n{question}\n"
        )
        return self._call(prompt, image_path, temperature=0.0)

    def pointwise(
        self,
        image_path: str,
        question: str,
        response: str,
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        prompt = (
            f"{POINTWISE_RUBRIC}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CANDIDATE RESPONSE:\n{response}\n"
        )
        return self._call(prompt, image_path, temperature)

    def listwise(
        self,
        image_path: str,
        question: str,
        candidates: Dict[str, str],
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        body = "\n\n".join(
            f"{label}:\n{text}" for label, text in candidates.items()
        )
        prompt = (
            f"{LISTWISE_RUBRIC}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CANDIDATES:\n{body}\n"
        )
        return self._call(prompt, image_path, temperature)

    def pairwise(
        self,
        image_path: str,
        question: str,
        response_a: str,
        response_b: str,
        model_label: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        prompt = (
            f"{PAIRWISE_RUBRIC}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"RESPONSE A:\n{response_a}\n\n"
            f"RESPONSE B:\n{response_b}\n"
        )
        return self._call(prompt, image_path, temperature)
