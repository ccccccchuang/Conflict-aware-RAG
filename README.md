# Conflict-Aware RAG: Multi-Stage Learning with Conflict Signals for Robust Retrieval-Augmented Generation

## Overall Framework

![Conflict-Aware RAG framework](images/conflict_aware_RAG.png)

**Conflict-Aware RAG** is a unified framework that leverages **ConScore**—a conflict scoring metric derived from the LLM itself—to guide data selection and training across SFT, DPO, and Reranking stages. By consistently filtering and organizing data with ConScore, the framework integrates model preferences into each stage, strengthening conflict modeling, improving robustness, and achieving better alignment between generation and reranking.

For the LLM training, we use the LLaMA-Factory framework to conduct SFT and DPO.

## Errata

We acknowledge two typographical errors in the published paper:

1. HotpotQA citation. The citation for HotpotQA is incorrect. The correct reference is:  
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

2. Equation (13). As described in the paper ("We optimize the reranker by minimizing the KL divergence, aligning its output distribution with the LLM's conflict-aware distribution."), the reranker is trained to match the LLM distribution, so the correct objective is the forward KL divergence, i.e., **KL(Q‖P)**. This is a typographical error in the manuscript only. The implementation is correct, and all reported experimental results remain unchanged.