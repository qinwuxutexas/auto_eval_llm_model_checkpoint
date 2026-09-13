# Auto Eval LLM Model Checkpoint Selection

Code and notebooks for **automated multimodal LLM checkpoint selection** using LLM-as-a-judge style evaluation (pointwise / listwise / preference-style ranking).

## Hugging Face mirror

Full package (Python package + eval data/images + demo/results) from:

**[vlmgrounding/AgenticCheckpointEval](https://huggingface.co/datasets/vlmgrounding/AgenticCheckpointEval/)**

Local copy in this repo: `AgenticCheckpointEval/`

| Path | Description |
|------|-------------|
| `AgenticCheckpointEval/src/agentic_eval/` | Core library (pointwise, listwise, pairwise, judges, pipeline, metrics) |
| `AgenticCheckpointEval/data/` | Evaluation images / inputs |
| `AgenticCheckpointEval/demo/` | Demo figures (Qwen2.5-VL experiment) |
| `AgenticCheckpointEval/results/` | Saved stage outputs and qualitative demos |

License on the HF dataset: Apache-2.0.

## Notebooks & docs (also in this repo)

| Path | Description |
|------|-------------|
| `auto_eval_llm_model_checkpoint.ipynb` | Main Colab notebook for checkpoint auto-evaluation |
| `auto_eval_llm_model_checkpoint_ipynb.py` | Python export of the main notebook |
| `agentic_checkpoint_eval.ipynb` | Agentic checkpoint evaluation notebook |
| `system_prompt.md` | Judge / system prompt used in evaluation |
| `README_qwen_case.md` | Notes for the Qwen case study |
| `docs/qwen25vl_7b_training_parameters_and_data.md` | Qwen2.5-VL-7B training parameters and data notes |

## Quick start

1. Prefer the packaged code under `AgenticCheckpointEval/src/agentic_eval/`, or open `auto_eval_llm_model_checkpoint.ipynb` in Colab/Jupyter.
2. Set your API key (do **not** commit secrets).
3. Run scoring / ranking stages and inspect outputs under `AgenticCheckpointEval/results/`.

## Notes

- Prefer re-downloading large assets from Hugging Face if you only need a subset.
- Do not commit API keys or private checkpoint weights.
