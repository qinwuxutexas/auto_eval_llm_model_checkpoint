Demo of checkpoint evaluation on the MetaCLIP + Llama-3-70B model.

Stage I and Stage III use Claude Sonnet 4.1 as the judge, while Stage II combines GPT-4.1 and Claude Sonnet 4.1 for listwise evaluation.

We present several challenging evaluation examples to illustrate:

The impact of sample quality on checkpoint selection. For example, images containing unreadable OCR often make checkpoints difficult to distinguish, while weaker checkpoints tend to produce more hallucinations.
The benefit of the complete multi-stage selection framework. Progressive sampling, statistical confidence estimation, and staged ranking together enable more reliable identification of the best checkpoint than a single-stage evaluation.