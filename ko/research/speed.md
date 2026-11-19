---
layout: research
lang: ko
ref: speed
pub: speed2026
permalink: /ko/research/speed/
locale: ko_KR
title: SPEED
description: "SPEED(Shallow Prefill, dEEp Decode)는 긴 프롬프트를 LLM의 하위층에만 남기고 생성 토큰은 모든 층을 쓰게 해, 벤치마크 점수를 거의 유지하면서 긴 문맥 추론 비용을 줄입니다."
---
<section id="tldr" class="rsec tldr" markdown="1">
## TL;DR

LLM은 긴 프롬프트를 읽을 때 모든 프롬프트 토큰의 key와 value(KV cache)를 모든 층에 저장하고, 토큰을 하나 생성할 때마다 이를 다시 읽습니다. SPEED는 프롬프트 토큰을 하위층에만 남기고, 모델이 생성하는 토큰은 그대로 모든 층을 거치게 합니다. Llama-3.1-8B에서 32층 가운데 24층에만 프롬프트를 남겼을 때 평균 점수는 51.2로, 전체 모델의 51.4와 비슷했습니다. 128K 문맥에서는 첫 토큰이 33%, 토큰당 디코딩이 22% 빨라졌고 KV 메모리는 25% 줄었습니다(Table 2).
</section>

<section id="problem" class="rsec" markdown="1">
## 긴 프롬프트가 비싼 이유

긴 문맥 추론에는 비용이 세 가지 듭니다. 프롬프트를 처리하는 prefill 때문에 첫 토큰이 늦게 나옵니다. 생성 단계에서는 토큰마다 저장된 key와 value를 읽어야 해서 메모리 대역폭이 병목이 됩니다. 그리고 cache 크기는 프롬프트 길이와 층 수에 함께 비례해 커집니다(1절).

기존 연구는 상위층 cache를 더 싸게 만드는 쪽이었습니다. 일부 토큰만 남기거나, cache를 압축·양자화하거나, 층끼리 상태를 공유하는 식입니다. SPEED는 더 단순하게 묻습니다. 전체 모델에서 생성 토큰은 중간층에서 프롬프트를 많이 보고, 상위층에서는 훨씬 덜 봅니다. 하위층이 이미 프롬프트에서 필요한 것을 가져간다면, 상위층의 프롬프트 KV가 꼭 있어야 할까요?
</section>

<section id="method" class="rsec" markdown="1">
## 방법

<figure class="figure figure--inline" tabindex="0" aria-label="SPEED 층 구조 그림">
{% include figures/speed.svg title="층과 토큰의 격자: 프롬프트 토큰은 K = 24층까지만 저장되고, BoS 토큰과 생성 토큰은 32층 모두에 남는 모습" %}
<figcaption>{{ site.data.i18n.ko.speed_fig_note }}</figcaption>
</figure>

SPEED는 L층 모델에서 기준 층 K를 정합니다(3절).

1. **Prefill.** 프롬프트 토큰은 1층부터 K층까지만 처리하고 저장합니다.
2. **Decode.** 생성 토큰은 L층을 모두 거칩니다. K층보다 위에서는 프롬프트가 아니라 앞서 생성한 토큰만 봅니다.
3. **Anchor.** 원래 있던 문장 시작(BoS) 토큰 하나만 모든 층에 남겨 기준점으로 씁니다. 학습된 요약 토큰이 아니라, 전체 깊이로 남겨 두는 유일한 프롬프트 토큰입니다.

모델 가중치, 학습 목표, 토큰 위치는 그대로이고, 어떤 KV를 만들어 두는지만 바뀝니다. 프롬프트가 N개 토큰이면 cache의 프롬프트 부분이 L·N개에서 K·N개로 줄어듭니다.

**학습.** 주 실험에서는 Llama-3.1-8B Base를 SPEED를 적용한 상태로 instruction tuning했습니다. 전체 깊이 기준 모델과 같은 178,502개 예시로 2 epoch 학습했습니다(4절). 이미 instruction tuning된 모델에 LoRA 1 epoch로 SPEED를 붙일 수도 있습니다.
</section>

<section id="results" class="rsec" markdown="1">
## 결과

{% include speed-results.html %}

- **Anchor가 중요합니다.** BoS anchor가 없으면 SPEED-24는 49.1까지 떨어집니다. 토큰 하나를 전체 깊이로 남기면 추가 비용 없이 51.2로 돌아오고, anchor가 없을 때 보이던 반복 루프도 줄어듭니다.
- **K는 조절 손잡이입니다.** K를 낮출수록 비용은 더 줄지만 품질도 떨어지고, K = 16에서는 45.4가 됩니다. 코드는 K를 크게 낮춰도 잘 버티는 반면, 수학과 지시 따르기가 가장 민감합니다.
- **Prefill뿐 아니라 디코딩도 빨라집니다.** K = 24에서 POP와 SwiftKV도 첫 토큰 속도는 비슷하게 높였지만, 저희 구현에서는 토큰당 디코딩 속도가 그대로였고 KV cache도 더 컸습니다. SPEED는 상위층이 프롬프트를 아예 읽지 않기 때문에 디코딩도 빨라집니다.
- **기존 모델에서 시작할 수 있습니다.** Llama-3.1-8B-Instruct에 LoRA를 1 epoch 적용한 SPEED-24는 HotpotQA에서 EM/F1 59.5/73.7을 기록했습니다. 같은 조건의 전체 깊이 LoRA는 60.8/75.3입니다(Table 3).

**한계.** 모든 결과는 32층 Llama-3.1-8B와 고정된 K에서 나왔습니다. 작은 점수 차이는 같은 조건의 비교일 뿐 동등성 검정이 아니므로, prefill을 얕게 해도 손실이 없다는 뜻은 아닙니다. POP·SwiftKV와는 효율만 비교했고, 실제 속도는 서빙 환경에 따라 달라집니다(6절).
</section>
