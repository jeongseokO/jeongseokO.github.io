---
layout: research
lang: en
ref: speed
pub: speed2026
permalink: /research/speed/
locale: en_US
title: SPEED
description: "SPEED (Shallow Prefill, dEEp Decode) keeps long prompts only in the lower layers of an LLM while generated tokens use the full depth, cutting long-context inference cost with little change in benchmark scores."
---
<section id="tldr" class="rsec tldr" markdown="1">
## TL;DR

When an LLM reads a long prompt, it stores keys and values (the KV cache) for every prompt token at every layer, and every generated token reads them again. SPEED keeps prompt tokens only in the lower layers, while the tokens the model generates still pass through the whole network. On Llama-3.1-8B, keeping the prompt in 24 of 32 layers scored 51.2 on average against 51.4 for the full model, with a 33% faster first token, 22% faster decoding per token, and 25% less KV memory at 128K context (Table 2).
</section>

<section id="problem" class="rsec" markdown="1">
## Why long prompts are expensive

Long-context inference pays three costs. Processing the prompt (prefill) delays the first token. Each generated token then reads the cached keys and values, so decoding is bound by memory bandwidth. And the cache grows with both the prompt length and the number of layers (Section 1).

Earlier work makes the upper-layer cache cheaper, by keeping only some tokens, compressing or quantizing the cache, or sharing states across layers. SPEED asks a blunter question. In a full model, generated tokens attend to the prompt heavily in the middle layers and much less in the upper layers. If the lower layers already take what they need from the prompt, does the model need upper-layer prompt KV at all?
</section>

<section id="method" class="rsec" markdown="1">
## How SPEED works

<figure class="figure figure--inline" tabindex="0" aria-label="SPEED layer diagram">
{% include figures/speed.svg title="Layer-by-token grid: prompt tokens are cached only up to layer K = 24, while the BoS token and generated tokens are kept in all 32 layers" %}
<figcaption>{{ site.data.i18n.en.speed_fig_note }}</figcaption>
</figure>

SPEED picks a cutoff layer K in an L-layer model (Section 3).

1. **Prefill.** Prompt tokens are processed and cached only in layers 1 to K.
2. **Decode.** Generated tokens go through all L layers. Above layer K they attend to earlier generated tokens, not to the prompt.
3. **Anchor.** The existing beginning-of-sequence (BoS) token stays in every layer as a stable reference point. It is not a learned summary, just the one prompt token kept at full depth.

The model's weights, training objective, and token positions are unchanged; only which KV states exist changes. For a prompt of N tokens, the prompt part of the cache shrinks from L·N to K·N entries.

**Training.** The main study instruction-tunes Llama-3.1-8B Base with SPEED applied during training, using the same 178,502 examples and two epochs as the full-depth baseline (Section 4). SPEED can also be added to an existing instruction model with one epoch of LoRA.
</section>

<section id="results" class="rsec" markdown="1">
## Results

{% include speed-results.html %}

- **The anchor matters.** Without the BoS anchor, SPEED-24 drops to 49.1. Keeping that single token at full depth brings it back to 51.2 at no extra cost, and it also suppresses the repetition loops seen in the anchor-free version.
- **Cutoff is a dial.** Lower K saves more but costs quality, and K = 16 falls to 45.4. Code holds up well even at aggressive cutoffs; math and instruction following are the most sensitive.
- **Faster decoding, not just faster prefill.** At K = 24, POP and SwiftKV give a similar first-token speedup, but in our implementation they do not speed up per-token decoding and keep a larger KV cache. SPEED does, because upper layers never read the prompt.
- **Works from an existing model.** With one epoch of LoRA on Llama-3.1-8B-Instruct, SPEED-24 reached 59.5/73.7 EM/F1 on HotpotQA, against 60.8/75.3 for full-depth LoRA (Table 3).

**Limits.** All results are on the 32-layer Llama-3.1-8B with fixed cutoffs. Small score gaps come from matched runs, not equivalence tests, so they do not show that shallow prefill is lossless. The POP and SwiftKV comparison covers efficiency only, and measured speed depends on the serving stack (Section 6).
</section>
