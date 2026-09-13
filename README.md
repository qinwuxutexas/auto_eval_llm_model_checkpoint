# Auto Eval LLM Model Checkpoint Selection

Code and notebooks for **automated multimodal LLM checkpoint selection** using LLM-as-a-judge style evaluation (pointwise / listwise / preference-style ranking).

## Contents

| Path | Description |
|------|-------------|
| `auto_eval_llm_model_checkpoint.ipynb` | Main Colab notebook for checkpoint auto-evaluation |
| `auto_eval_llm_model_checkpoint_ipynb.py` | Python export of the main notebook |
| `agentic_checkpoint_eval.ipynb` | Agentic checkpoint evaluation notebook |
| `system_prompt.md` | Judge / system prompt used in evaluation |
| `README_qwen_case.md` | Notes for the Qwen case study |
| `docs/qwen25vl_7b_training_parameters_and_data.md` | Qwen2.5-VL-7B training parameters and data notes |

## Quick start

1. Open `auto_eval_llm_model_checkpoint.ipynb` in Colab or Jupyter.
2. Set your API key (do **not** commit secrets).
3. Follow the notebook cells for scoring checkpoints and aggregating rankings.

## Notes

- This repository hosts the evaluation / selection codebase and prompts.
- Large qualitative demo image dumps are kept out of git by default.
- Original Colab source for the main notebook: see notebook metadata / header comment.

## License

Add a license of your choice if/when you open this for redistribution.
