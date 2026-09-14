"""Single source of truth for judge rubrics."""

POINTWISE_RUBRIC = """
Evaluate the candidate response to the image/question.

Dimensions:
1. Factual correctness and support from visible evidence.
2. Relevance to the user's question.
3. Usefulness/completeness without unnecessary content.
4. Visual/OCR grounding.
5. Hallucination: penalize unsupported claims.

Groundedness takes precedence over stylistic fluency.

Return strict JSON:
{
  "score": <float from 0 to 1>,
  "factuality": <0 to 1>,
  "relevance": <0 to 1>,
  "usefulness": <0 to 1>,
  "grounding": <0 to 1>,
  "hallucination_penalty": <0 to 1>,
  "rationale": "<brief>"
}
""".strip()


LISTWISE_RUBRIC = """
Rank all anonymous candidate responses to the SAME image/question.

Primary criteria:
1. factual/visual correctness,
2. relevance,
3. usefulness,
4. absence of hallucination,
5. OCR/visual grounding.

Do not infer checkpoint quality from candidate labels or ordering.

Return strict JSON:
{
  "ranking": ["C3", "C1", "C2", ...],
  "rationale": {"C3": "...", "C1": "...", ...}
}
""".strip()


PAIRWISE_RUBRIC = """
Choose which anonymous response better answers the question from the image.

Prioritize:
- factual/visual correctness,
- relevance,
- usefulness,
- OCR/visual grounding,
- absence of hallucination.

Return strict JSON:
{
  "winner": "A" or "B" or "TIE",
  "confidence": <0 to 1>,
  "rationale": "<brief>"
}
""".strip()


EVALUABILITY_RUBRIC = """
You are evaluating the quality of a multimodal VQA evaluation sample.

Evaluate ONLY the IMAGE and QUESTION.
Do NOT consider any candidate model response.

Your task is to determine whether the visual evidence REQUIRED TO
ANSWER THIS SPECIFIC QUESTION is sufficiently visible.

Return ONLY valid JSON:
{
  "visual_evaluability": 0.0,
  "required_evidence_readability": 0.0,
  "answerability": 0.0,
  "reason": ""
}
""".strip()
