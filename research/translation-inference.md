---
layout: research
lang: en
ref: translation
permalink: /research/translation-inference/
locale: en_US
title: Inference strategies for regulatory translation
eyebrow: Prompt Engineering
context: "Part of the MFDS translation project · SKI-ML Lab × Ministry of Food and Drug Safety · Feb 2025 – Nov 2026"
description: "How we prompted a Korean–English translation model for drug regulatory documents, and the post-editing loop that checks and repairs its output."
---
<section id="tldr" class="rsec tldr" markdown="1">
## TL;DR

In the MFDS translation project, how we asked the model to translate mattered almost as much as how we trained it. We compared four prompt templates and kept two: retrieval few-shot, which was the most accurate, and plain segment translation, which was the fastest. An LLM will still produce broken output now and then, so we also wrapped the model in a post-editing loop that checks every translation and repairs or flags the ones that fail.
</section>

<section id="templates" class="rsec" markdown="1">
## Choosing a prompt template

| Template | What the model sees |
|---|---|
| Segment | Only the sentence to translate |
| Source Context & Last | The surrounding source text and the previous translation |
| Context Parallel (single / multi-turn) | Neighboring source–translation pairs, as one turn or several |
| Retrieval Few-shot | Translation pairs from a database that are similar to the input |

We compared them on accuracy (BLEU) and speed. Retrieval Few-shot was the most accurate (BLEU 62.3) while staying fast. Segment gave up a little accuracy (BLEU 58.7) but was the fastest, so we kept both as final templates.
</section>

<section id="loop" class="rsec" markdown="1">
## A safety net around the model

When we read the fine-tuned model's output by hand, quality was good overall, but the model sometimes repeated itself or broke the format, and serious mistakes showed up in as much as about 5% of outputs. We grouped the recurring problems into four checks:

| Check | What fails it |
|---|---|
| Length | The translation is far longer or shorter than the source |
| Placeholder | A number or symbol changes, e.g. "6.5.2. Pivotal studies" coming back as section 6.5.3 |
| Copy | The source sentence comes back untranslated |
| QE | A quality-estimation model (CometKiwi) scores the translation low |

<figure class="figure figure--inline" tabindex="0" aria-label="Post-editing loop diagram">
{% include figures/loop.svg title="Four-step loop: translate, re-translate with other templates, post-edit, then select the top candidates and warn the user; a translation goes to the output as soon as it passes the checks" %}
</figure>

1. Translate with greedy decoding and run the checks.
2. If the translation fails, generate candidates with other prompt templates and check those.
3. If they fail too, revise them with a post-editing model and check again.
4. If nothing passes, return the top candidates ranked by risk and QE score, and show the user which check failed.

In our tests the loop lowered the rate of detected errors, and the output stayed normal even when we fed it wrong words on purpose. The checks are rule-based, so they also raise some false alarms.
</section>
