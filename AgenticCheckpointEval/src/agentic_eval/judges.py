
import base64
import json
import re
from pathlib import Path

from openai import OpenAI


class OpenAIJudgeClient:

    def __init__(
        self,
        api_key,
        model="gpt-4o",
        repo_root="/content/AgenticCheckpointEval_git",
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.repo_root = Path(repo_root)

    # ============================================================
    # IMAGE
    # ============================================================

    def _encode_image(self, image_path):

        path = Path(image_path)

        if not path.is_absolute():
            path = self.repo_root / path

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "rb") as f:
            return base64.b64encode(
                f.read()
            ).decode("utf-8")

    def _parse_json(self, text):

        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.I,
        ).strip()

        return json.loads(text)

    # ============================================================
    # IMAGE-QUESTION EVALUABILITY
    # ============================================================

    def evaluability(
        self,
        image_path,
        question,
    ):

        image_b64 = self._encode_image(image_path)

        prompt = f"""
You are evaluating the quality of a multimodal VQA evaluation sample.

Evaluate ONLY the IMAGE and QUESTION.
Do NOT consider any candidate model response.

QUESTION:
{question}

Your task is to determine whether the visual evidence REQUIRED TO
ANSWER THIS SPECIFIC QUESTION is sufficiently visible.

Evaluate:

1. visual_evaluability [0,1]

Could an evaluator reliably determine whether an answer to THIS
QUESTION is correct using this image?

1.0 = clearly and reliably evaluable
0.0 = impossible or highly unreliable to evaluate


2. required_evidence_readability [0,1]

How clearly is the specific visual evidence required by THIS QUESTION
visible or readable?

For OCR-dependent questions:
evaluate readability of the PARTICULAR text required to answer the
question, rather than text in the image generally.

For non-OCR questions:
evaluate visibility of the required objects, attributes,
relationships, actions, or scene information.

1.0 = required evidence clearly visible
0.0 = required evidence unreadable, invisible, occluded, blurred,
too small, cropped, or otherwise unavailable


3. answerability [0,1]

Does the image itself contain sufficient visual evidence to answer
THIS QUESTION?

1.0 = sufficient evidence
0.0 = insufficient evidence


IMPORTANT:

- Evaluate the IMAGE and QUESTION jointly.
- Do NOT use or imagine candidate responses.
- Do NOT reconstruct unreadable text using world knowledge.
- Do NOT infer missing evidence from likely context.
- Do NOT penalize unreadable text that is irrelevant to the question.
- A visually poor image may still be highly evaluable if the evidence
  required by the question is clearly visible.
- A generally clear image may be poorly evaluable if the particular
  evidence required by the question is unreadable.

Return ONLY valid JSON:

{{
  "visual_evaluability": 0.0,
  "required_evidence_readability": 0.0,
  "answerability": 0.0,
  "reason": ""
}}
"""

        result = self.client.responses.create(
            model=self.model,
            temperature=0.0,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url":
                                f"data:image/png;base64,{image_b64}",
                        },
                    ],
                }
            ],
        )

        return self._parse_json(
            result.output_text
        )

    # ============================================================
    # POINTWISE CHECKPOINT JUDGMENT
    # ============================================================

    def pointwise(
        self,
        image_path,
        question,
        response,
        checkpoint=None,
    ):

        image_b64 = self._encode_image(image_path)

        prompt = f"""
You are evaluating a multimodal assistant response.

Inspect the IMAGE carefully.

QUESTION:
{question}

CANDIDATE RESPONSE:
{response}

Evaluate the candidate using the evidence actually visible in the
image.

Score each dimension from 0 to 4:

1. factuality
Is the response factually supported by the image?

2. relevance
Does the response directly answer the question?

3. usefulness
Is the answer useful and appropriately informative?

4. hallucination
Does the response avoid unsupported visual or textual claims?

For hallucination:
4 = no unsupported claims
0 = severe unsupported/hallucinated claims

IMPORTANT:

- Do not reward plausible information that cannot be verified.
- OCR/text claims must be visually supported.
- Do not reconstruct unreadable text from world knowledge.
- Ignore checkpoint identity.
- Judge the response itself rather than writing style.

Also provide:

confidence [0,1]:
Your confidence that THIS EVALUATION is reliable.
This is NOT the probability that the candidate answer is correct.

Return ONLY valid JSON:

{{
  "factuality": 0,
  "relevance": 0,
  "usefulness": 0,
  "hallucination": 0,
  "score": 0.0,
  "confidence": 0.0,
  "rationale": ""
}}

score must be normalized to [0,1].
"""

        result = self.client.responses.create(
            model=self.model,
            temperature=0.0,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url":
                                f"data:image/png;base64,{image_b64}",
                        },
                    ],
                }
            ],
        )

        return self._parse_json(
            result.output_text
        )


class MockJudgeClient:

    def pointwise(
        self,
        image_path,
        question,
        response,
        checkpoint=None,
    ):

        import hashlib

        key = (
            str(question)
            + str(response)
            + str(checkpoint)
        )

        h = int(
            hashlib.md5(
                key.encode()
            ).hexdigest()[:8],
            16,
        )

        score = (h % 1000) / 1000.0

        return {
            "factuality": round(score * 4),
            "relevance": round(score * 4),
            "usefulness": round(score * 4),
            "hallucination": round(score * 4),
            "score": score,
            "confidence": 0.5,
            "rationale": "Mock evaluation.",
        }
