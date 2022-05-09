# Entropy-based Fuzzy SVM 구현하기

## 개요
- EFSVM은 imbalanced dataset을 분류할 때 유용하게 사용할 수 있는 머신러닝 기법이다. 
- 일반적인 SVM 앞에 추가적인 장치를 적용해서 dataset 원본을 해치지 않고 imbalanced dataset을 handling할 수 있는 획기적인 방법이다.
- 파이선을 이용해서 EFSVM을 구현해보자.

## EFSVM이란?

- SVM의 decision surface를 결정하는데 negative class의 영향력을 축소시키는 것을 목적으로 설계된 모형이다.
- 각 데이터의 중요도를 Entropy를 사용해서 나타낸다. 
![image-20220509235512284](https://raw.githubusercontent.com/jeongseokO/jeongseokO.github.io/assets/images/assets/images/image-20220509235512284.png)



```python

```
