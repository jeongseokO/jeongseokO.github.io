---
layout: research
lang: ko
ref: translation
permalink: /ko/research/translation-inference/
locale: ko_KR
title: 의약품 규제 문서 번역의 추론 전략
eyebrow: Prompt Engineering
context: "식약처 번역 과제의 일부 · SKI-ML 연구실 × 식품의약품안전처 · 2025.02 – 2026.11"
description: "의약품 규제 문서용 한-영 번역 모델에 프롬프트를 어떻게 줬는지, 그리고 출력을 검사하고 고치는 후편집 루프를 소개합니다."
---
<section id="tldr" class="rsec tldr" markdown="1">
## TL;DR

식약처 번역 과제에서는 모델을 어떻게 학습하느냐만큼, 번역을 어떻게 시키느냐도 중요했습니다. 프롬프트 템플릿 네 가지를 비교해서 가장 정확한 Retrieval Few-shot과 가장 빠른 Segment 두 가지를 남겼습니다. 그래도 LLM은 가끔 잘못된 출력을 내기 때문에, 모든 번역을 검사하고 통과하지 못한 번역은 고치거나 경고를 붙이는 후편집 루프로 모델을 감쌌습니다.
</section>

<section id="templates" class="rsec" markdown="1">
## 프롬프트 템플릿 고르기

| 템플릿 | 모델에 주는 정보 |
|---|---|
| Segment | 번역할 문장만 |
| Source Context & Last | 주변 원문과 직전 번역 |
| Context Parallel (single / multi-turn) | 앞뒤 원문–번역 쌍을 한 턴 또는 여러 턴으로 |
| Retrieval Few-shot | 데이터베이스에서 찾은, 입력과 비슷한 번역 예시 |

정확도(BLEU)와 속도로 비교했습니다. Retrieval Few-shot은 가장 정확하면서도(BLEU 62.3) 빨랐습니다. Segment는 정확도가 조금 낮았지만(BLEU 58.7) 가장 빨라서, 두 가지를 최종 템플릿으로 정했습니다.
</section>

<section id="loop" class="rsec" markdown="1">
## 모델을 감싸는 안전장치

미세조정한 모델의 출력을 직접 읽어 보니 전반적인 품질은 좋았지만, 같은 말을 반복하거나 형식이 깨지는 경우가 있었고 심각한 실수가 많게는 5% 정도 나왔습니다. 자주 나오는 문제를 네 가지 검사로 묶었습니다.

| 검사 | 걸리는 경우 |
|---|---|
| Length | 번역문이 원문보다 지나치게 길거나 짧을 때 |
| Placeholder | 숫자나 기호가 바뀔 때. 예: "6.5.2. Pivotal studies"가 6.5.3절로 바뀜 |
| Copy | 원문이 번역되지 않고 그대로 나올 때 |
| QE | 품질 평가 모델(CometKiwi)이 낮은 점수를 줄 때 |

<figure class="figure figure--inline" tabindex="0" aria-label="후편집 루프 그림">
{% include figures/loop.svg title="네 단계 루프: 번역, 다른 템플릿으로 재번역, 후편집, 상위 후보 선택과 경고. 검사를 통과하면 바로 출력으로 나감" %}
</figure>

1. Greedy decoding으로 번역하고 검사합니다.
2. 통과하지 못하면 다른 프롬프트 템플릿으로 후보를 만들어 다시 검사합니다.
3. 그래도 안 되면 후편집 모델로 후보를 고친 뒤 다시 검사합니다.
4. 끝까지 통과하는 후보가 없으면 위험도와 QE 점수로 상위 후보를 골라 돌려주고, 어떤 검사에서 걸렸는지 사용자에게 보여 줍니다.

실험에서 이 루프를 거치면 검출되는 오류 비율이 줄었고, 일부러 잘못된 단어를 넣어도 출력이 정상적으로 유지됐습니다. 검사가 규칙 기반이라 오탐도 일부 있습니다.
</section>
