---
layout: research
lang: ko
ref: lsc
pub: lsc2026
permalink: /ko/research/lsc/
locale: ko_KR
title: Latent Self-Consistency
description: "Latent Self-Consistency(AAAI 2026): 학습된 summary token 임베딩으로 여러 LLM 응답 가운데 가장 일관된 응답을 고르는 방법. 짧은 답과 긴 답 모두에 적용되며 추론 시간은 1% 미만으로 늘어납니다."
---
<section id="tldr" class="rsec tldr" markdown="1">
## 요약

여러 응답을 샘플링한 뒤 다수결로 답을 고르는 Self-Consistency(SC)는 LLM의 정확도를 높이지만, 답을 문자열 그대로 비교할 수 있을 때만 동작합니다. Latent Self-Consistency(LSC)는 대신 학습된 summary token 임베딩으로 응답을 비교합니다. 그래서 숫자, 객관식 보기, 코드, 자유 형식 글에 같은 선택 규칙을 쓸 수 있습니다. 세 모델에서 LSC는 짧은 답 평균 정확도가 SC와 0.3%p 이내로 같은 수준이고, SC를 적용할 수 없는 긴 답 과제까지 처리하며, 추론 시간은 최대 0.9%만 늘어납니다(Table 1).
</section>

<section id="problem" class="rsec" markdown="1">
## 문제

Self-Consistency는 응답 N개를 샘플링하고 가장 자주 나온 최종 답을 돌려줍니다. 답을 세려면 문자열이 정확히 일치해야 하므로, 코드나 요약, 설명처럼 같은 뜻을 다르게 표현하는 답에서는 SC가 동작하지 않습니다.

긴 답을 다루는 기존 확장 방법이 두 가지 있지만, 각각 대가가 있습니다. Universal Self-Consistency(USC)는 LLM에게 어떤 응답이 가장 일관된지 판단하게 하며, 논문에 따르면 추론 시간이 약 10%, 메모리가 약 15% 더 듭니다. Weighted Unigram Consistency Score(WUCS)는 단어 겹침을 가볍게 비교하지만 짧은 답 과제에서 정확도가 떨어집니다. 질문마다 답 형식에 맞는 방법을 골라 쓰는 것도 해결책이 되지 못합니다. 논문은 질문만 보고는 모델이 짧게 답할지 길게 답할지 안정적으로 예측하기 어렵다는 것을 확인했습니다(Appendix K).
</section>

<section id="method" class="rsec" markdown="1">
## 방법

<figure class="figure">
  <div class="figure__scroll" tabindex="0" role="region" aria-label="방법 개요 그림">
    <img src="{{ '/assets/img/lsc-overview.svg' | relative_url }}" alt="LSC의 세 단계: 고정된 기반 LLM이 응답 N개를 샘플링하고, 각 응답의 EOS 토큰 뒤에 붙인 학습 가능한 summary token이 KV cache를 재사용해 응답마다 임베딩 하나를 만들며, 다른 응답들과 가장 비슷한 응답을 고르고 동떨어진 응답은 무시합니다." width="760" height="332">
  </div>
  <figcaption>{{ site.data.i18n.ko.fig_note }}</figcaption>
</figure>

1. **후보 생성.** 기반 LLM이 질문에 대한 응답 N개를 샘플링합니다.
2. **생성 후 인코딩.** 각 응답의 EOS(end-of-sequence) 토큰 뒤에 학습 가능한 summary token K개를 붙입니다. 생성할 때 만든 key–value cache를 재사용하므로, 모델은 새로 붙인 K개 위치만 계산합니다. 이 토큰들의 마지막 층 표현을 평균하고 정규화해 응답마다 임베딩 *z<sub>i</sub>* 하나를 얻습니다.
3. **다수 응답 선택.** 모든 임베딩 쌍의 코사인 유사도 *S<sub>ij</sub>*를 계산하고, 각 응답의 점수를 *w<sub>i</sub>* = (1/(N−1)) Σ<sub>j≠i</sub> exp(*S<sub>ij</sub>* / τ′)로 매깁니다(τ′ = 0.5). 지수 가중치를 쓰면 동떨어진 응답 몇 개가 모든 점수를 평균 쪽으로 끌어당기는 일을 막을 수 있습니다. 점수가 가장 높은 응답을 돌려줍니다. 선택 사항인 Dynamic Top-K는 유사도 구조에서 다수 집합의 크기를 추정하고, 각 응답을 가장 가까운 이웃들하고만 비교해 점수를 매깁니다.

**학습.** summary token의 임베딩만 학습합니다. 기반 LLM은 고정되어 있으므로 모델이 생성하는 응답은 바뀌지 않습니다. 8B 모델에서 K = 6이면 전체 파라미터의 3 × 10<sup>−6</sup> 미만만 업데이트합니다. 학습 신호는 supervised contrastive loss(τ = 0.07)입니다. 같은 답에 도달한 응답끼리는 가깝게, 다른 답에 도달한 응답끼리는 멀게 만듭니다. 학습 데이터는 짧은 답과 긴 답 샘플을 50:50으로 섞었고, GSM8K, MATH, TriviaQA, MMLU에서 LLM 의사 레이블(pseudo-label)로 만들었습니다.
</section>

<section id="results" class="rsec" markdown="1">
## 결과

**실험 설정.** 질문마다 nucleus sampling(top-p 0.95, temperature 0.9)으로 후보 10개를 생성합니다. 모델은 LLaMA3.1-8B-Instruct, Qwen3-8B, LLaMA3.3-70B-Instruct입니다. 짧은 답 벤치마크 6개(GSM8K, MATH, TriviaQA, MMLU, CommonsenseQA, TruthfulQA-MC1)와 긴 답 벤치마크 5개(TruthfulQA, MSMARCO-NLG, HumanEval/HumanEval+, MBPP/MBPP+, CNN/DailyMail)로 평가합니다. 긴 답 벤치마크는 학습에 전혀 쓰지 않았습니다.

{% include lsc-results.html %}

- **짧은 답.** 세 모델 모두 LSC의 평균이 SC와 0.3%p 이내입니다(+0.1, −0.3, −0.1).
- **긴 답.** 두 LLaMA 모델에서는 LSC가 모든 기존 방법보다 높습니다(65.2 대 USC 64.8, 77.6 대 USC 76.8). Qwen3-8B에서는 WUCS가 더 높습니다(75.4 대 74.9).
- **비용.** USC는 시간이 7.4–18.2%, 메모리가 7.5–20.2% 더 듭니다. LSC는 시간이 최대 0.9%, 메모리가 최대 0.005% 더 듭니다.

**일관성.** 논문은 선택된 응답이 후보 10개 가운데 실제 다수 집합에 속하는 비율도 측정했습니다. 벤치마크 7개 평균(Table 4)에서 LSC는 세 모델에서 각각 93.0%, 96.1%, 97.3%로, USC의 79.1%, 84.4%, 90.1%보다 높습니다.

**보정(calibration).** LSC의 유사도 점수는 신뢰도 추정치로 쓸 수 있습니다. 기대 보정 오차(ECE, Table 5)는 MATH에서 0.140, TriviaQA에서 0.052로, SC의 0.076과 0.044보다 큽니다. 즉 짧은 답에서는 SC가 더 잘 보정되어 있습니다. SC를 적용할 수 없는 긴 답에서는 LSC의 오차(TruthfulQA 0.217, HumanEval 0.170)가 WUCS(0.349, 0.316)보다 작습니다.
</section>
