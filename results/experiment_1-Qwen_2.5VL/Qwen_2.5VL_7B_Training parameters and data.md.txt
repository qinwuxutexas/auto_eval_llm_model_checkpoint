Training Configuration
The evaluated checkpoints were produced by supervised fine-tuning (SFT) of Qwen2.5-VL-7B-Instruct using LoRA. Training used a single GPU with bf16 precision, an effective batch size of 8, cosine learning-rate decay (initial learning rate 2×10⁻⁴), and one epoch over 10,000 training samples (~1250 optimization steps). Approximately ten checkpoints were saved during training at 125-step intervals for subsequent evaluation. LoRA was applied to q_proj, k_proj, v_proj, and o_proj with rank 64, α=128, and dropout 0.05. The held-out WearVQA benchmark was used for checkpoint evaluation. 

Training Data
The SFT corpus consists of 10,000 image-question-answer pairs collected from multiple public vision-language benchmarks with random selection.

Dataset	Samples	Share
VQAv2	2,000	20%
VizWiz	1,500	15%
TextVQA	1,500	15%
OCR-VQA	1,500	15%
A-OKVQA	1,000	10%
DocVQA	800	8%
ChartQA	800	8%
OK-VQA	500	5%
ScienceQA	400	4%
Total	10,000	100%

Evaluation dataset: https://huggingface.co/datasets/tonyliao-meta/WearVQA


Agentic Checkpoint Evaluation
Candidate checkpoints are evaluated using a three-stage pipeline that progressively increases evaluation fidelity while reducing computational cost. Stage I independently scores each checkpoint on every sample and filters weaker candidates. Stage II jointly ranks the remaining checkpoints using listwise evaluation with Borda aggregation. Stage III performs pairwise comparison between the two finalists and estimates selection uncertainty using bootstrap resampling. :contentReference[oaicite:1]{index=1}

Candidate checkpoints
        │
        ▼
Stage I
Pointwise Evaluation
        │
   Keep Top-k
        │
        ▼
Stage II
Listwise Ranking
(Borda Aggregation)
        │
 Top-2 Finalists
        │
        ▼
Stage III
Pairwise Comparison
(Win/Loss/Tie)
        │
 Bootstrap Stability
        │
        ▼
Selected Checkpoint