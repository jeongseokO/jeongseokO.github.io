---
title:  "**Entropy-based Fuzzy SVM 구현하기**"
excerpt: "EFSVM 구현"

categories:
 - reproduce
 - theory
tags:
 - [SVM]

toc: true
toc_sticky: true

date: 2022-05-10
last_modified_at: 2022-05-10

published: true
---

## 개요
- EFSVM은 imbalanced dataset을 분류할 때 유용하게 사용할 수 있는 머신러닝 기법이다. 
- 일반적인 SVM 앞에 추가적인 장치를 적용해서 dataset 원본을 해치지 않고 imbalanced dataset을 handling할 수 있는 획기적인 방법이다.
- 파이선을 이용해서 EFSVM을 구현해보자.

## EFSVM이란?

- SVM의 decision surface를 결정하는데 negative class의 영향력을 축소시키는 것을 목적으로 설계된 모형이다.
- 각 데이터의 중요도를 Entropy를 사용해서 나타낸다. 
  ![image-20220510001312617](/assets/images/image-20220510001312617.png)
- K는 각 데이터의 Nearest Neighbor의 파라미터이다. 
  ![image-20220510002546213](/assets/images/image-20220510002546213.png)
- K를 결정해서 KNN을 진행한 후, 그 내부에서 엔트로피를 계산한다. 
- 샘플의 수가 적은 class를 positive, 많은 쪽을 negative라고 둔다. Negative class에 대해서만 엔트로피를 사용하여 중요도를 낮추어줄 것이다. 
- 엔트로피 H의 최대, 최소를 구하고, 적절한 수의 Membership subset들을 만들어준다. 이때 Membership들의 엔트로피는 작은 것부터 큰 순서로 정렬해준다. 
  ![image-20220510003114244](/assets/images/image-20220510003114244.png)
- 위 식과 같이 각 Fuzzy Membership별로 중요도를 낮춰준다. 이때 베타는 다음과 같은 지침을 갖는 하이퍼 파라미터이다. 
  ![image-20220510003227933](/assets/images/image-20220510003227933.png)
- 베타가 0에 가까울수록 Positive와 Negative class의 중요도에 대한 차별성이 떨어진다.
- 모든 데이터에 적용하면 다음과 같은 결과를 얻을 수 있다.
  ![image-20220510003243938](/assets/images/image-20220510003243938.png)


- 다음 식을 보면 우리가 구한 s가 어떤 역할을 하는지 바로 알 수 있다.
  ![image-20220510003523722](/assets/images/image-20220510003523722.png)
- s는 Soft margin SVM의 slack variable에 곱해져서 해당 데이터의 영향력을 낮추는 역할을 한다. 이 효과를 오직 negative class의 데이터에만 적용하여 imbalance dataset을 극복하는 것이 이 모델의 핵심이다. 


```python

```
