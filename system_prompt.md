
POINTWISE_RUBRIC = """
Evaluate the candidate response to the image and question.

Evaluation dimensions:
1. Factual correctness and support from visible evidence.
2. Relevance to the user's question.
3. Usefulness and completeness without unnecessary content.
4. Visual/OCR grounding.
5. Hallucination: penalize unsupported claims.

Factual and visual grounding takes precedence over stylistic fluency.
Return strict JSON:
{
    "score": <float from 0 to 1>,
    "factuality": <float from 0 to 1>,
    "relevance": <float from 0 to 1>,
    "usefulness": <float from 0 to 1>,
    "grounding": <float from 0 to 1>,
    "hallucination_penalty": <float from 0 to 1>,
    "rationale": "<brief explanation>"
}
"""

LISTWISE_RUBRIC = """
Rank all anonymous candidate responses to the SAME image and question.
Primary criteria:
1. Factual and visual correctness.
2. Relevance.
3. Usefulness.
4. Absence of hallucination.
5. OCR and visual grounding.
Do not infer checkpoint quality from candidate labels or presentation order.
Return strict JSON:
{
    "ranking": ["C3", "C1", "C2"],
    "rationale": {
        "C3": "...",
        "C1": "...",
        "C2": "..."
    }
}
"""

PAIRWISE_RUBRIC = """
Compare two anonymous responses A and B to the same image and question.
Prioritize:
1. Factual and visual correctness.
2. Relevance.
3. Usefulness.
4. OCR and visual grounding.
5. Absence of hallucination.
Return strict JSON:
{
    "winner": "A" or "B" or "TIE",
    "confidence": <float from 0 to 1>,
    "rationale": "<brief explanation>"
}
"""
