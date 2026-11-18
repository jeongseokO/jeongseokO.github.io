---
layout: research
lang: en
ref: lsc
pub: lsc2026
permalink: /research/lsc/
locale: en_US
title: Latent Self-Consistency
description: "Latent Self-Consistency (AAAI 2026): selecting the most consistent of several LLM responses with learned summary-token embeddings, for both short and long answers, at under 1% extra inference time."
---
<section id="tldr" class="rsec tldr" markdown="1">
## TL;DR

Sampling several responses and keeping the majority answer, known as Self-Consistency (SC), makes LLMs more accurate, but it only works when answers can be compared as exact strings. Latent Self-Consistency (LSC) compares responses through learned summary-token embeddings instead, so one selection rule works for numbers, multiple-choice letters, code, and free-form text. Across three models, LSC stays within 0.3 points of SC's short-answer average, extends to long-answer tasks where SC cannot be applied, and adds at most 0.9% to inference time (Table 1).
</section>

<section id="problem" class="rsec" markdown="1">
## The problem

Self-Consistency samples N responses and returns the most frequent final answer. Counting answers requires exact string matches, so SC breaks down when equivalent answers are worded differently, as in code, summaries, or explanations.

Two existing extensions handle long answers, but each gives something up. Universal Self-Consistency (USC) asks the LLM itself to judge which response is most consistent; the paper reports about 10% extra inference time and about 15% extra memory for this. Weighted Unigram Consistency Score (WUCS) compares word overlap cheaply, but loses accuracy on short-answer tasks. Routing each query to a format-specific method is not a way out either: the paper finds that the question alone does not reliably predict whether the model will answer briefly or at length (Appendix K).
</section>

<section id="method" class="rsec" markdown="1">
## How LSC works

<figure class="figure">
  <div class="figure__scroll" tabindex="0" role="region" aria-label="Method overview figure">
    <img src="{{ '/assets/img/lsc-overview.svg' | relative_url }}" alt="LSC in three stages: the frozen base LLM samples N responses; learnable summary tokens appended after each response's EOS token turn it into one embedding, reusing the KV cache; the response most similar to the others is selected and an outlier is ignored." width="760" height="332">
  </div>
  <figcaption>{{ site.data.i18n.en.fig_note }}</figcaption>
</figure>

1. **Candidate generation.** The base LLM samples N responses to the question.
2. **Post-generation encoding.** K learnable summary tokens are appended after each response's end-of-sequence token. Because the key–value cache from generation is reused, the model computes only the K new positions. Their final-layer states are averaged and normalized into one embedding *z<sub>i</sub>* per response.
3. **Majority selection.** LSC computes the cosine similarity *S<sub>ij</sub>* between every pair of embeddings and scores each response by *w<sub>i</sub>* = (1/(N−1)) Σ<sub>j≠i</sub> exp(*S<sub>ij</sub>* / τ′), with τ′ = 0.5. The exponential weighting keeps a few outliers from pulling every score toward the average. The response with the highest score is returned. An optional Dynamic Top-K step estimates the size of the majority set from the similarity structure and scores each response only against its nearest neighbors.

**Training.** Only the summary-token embeddings are trained. The base LLM stays frozen, so the responses it generates are unchanged. With K = 6 on an 8B model, this updates fewer than 3 × 10<sup>−6</sup> of the parameters. The training signal is a supervised contrastive loss (τ = 0.07): responses that reach the same answer are pulled together, and responses with different answers are pushed apart. The training data mixes short- and long-answer samples 50:50, built with LLM pseudo-labels from GSM8K, MATH, TriviaQA, and MMLU.
</section>

<section id="results" class="rsec" markdown="1">
## Results

**Setup.** Each question gets 10 candidates from nucleus sampling (top-p 0.95, temperature 0.9). The models are LLaMA3.1-8B-Instruct, Qwen3-8B, and LLaMA3.3-70B-Instruct. Six benchmarks have short answers (GSM8K, MATH, TriviaQA, MMLU, CommonsenseQA, TruthfulQA-MC1) and five have long answers (TruthfulQA, MSMARCO-NLG, HumanEval/HumanEval+, MBPP/MBPP+, CNN/DailyMail). None of the long-answer benchmarks was used for training.

{% include lsc-results.html %}

- **Short answers.** LSC's average is within 0.3 points of SC on all three models (+0.1, −0.3, −0.1).
- **Long answers.** LSC is above every baseline on both LLaMA models (65.2 vs. 64.8 for USC; 77.6 vs. 76.8 for USC). On Qwen3-8B, WUCS is higher (75.4 vs. 74.9).
- **Cost.** USC adds 7.4–18.2% time and 7.5–20.2% memory. LSC adds at most 0.9% time and at most 0.005% memory.

**Consistency.** The paper also measures how often the selected response belongs to the true majority among the 10 candidates. Averaged over seven benchmarks (Table 4), LSC reaches 93.0%, 96.1%, and 97.3% on the three models, compared with 79.1%, 84.4%, and 90.1% for USC.

**Calibration.** LSC's similarity score can serve as a confidence estimate. Its expected calibration error (Table 5) is 0.140 on MATH and 0.052 on TriviaQA, against 0.076 and 0.044 for SC, so SC remains better calibrated on short answers. On long answers, where SC cannot be applied, LSC's error (0.217 on TruthfulQA, 0.170 on HumanEval) is lower than that of WUCS (0.349 and 0.316).
</section>
