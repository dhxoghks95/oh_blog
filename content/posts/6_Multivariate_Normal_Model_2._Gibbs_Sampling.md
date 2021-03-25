---
title: A First Course In Bayesian Statistical Methods Chapter 6. Multivariate Normal Model, 2. Gibbs Sampling
date: 2021-03-25T18:22:38+09:00
categories: ["A First Course In Bayesian Statistical Methods"]
tags: ["Bayesian", "Python"]
---

# A First Course in Bayesian Statistical Methods

## 6. 다변량 정규 모델(Multivariate normal model)

## 6.4 깁스 샘플러를 사용한 평균과 공분산 표본 추출

지난 포스팅에서 다음을 배웠습니다.

$$
\{ \boldsymbol{\theta} | \mathbf{y_1}, ..., \mathbf{y_n}, \Sigma \} \sim MVN(\boldsymbol{\mu_n}, \Lambda_n) \\
\{ \Sigma | \mathbf{y_1}, ..., \mathbf{y_n}, \boldsymbol{\theta} \} \sim \text{inverse-Wishart}(\nu_n, \boldsymbol{S_n}^{-1})
$$

이 때 $\Lambda_n, \mu_n$은 식 (6.4), (6.5)에서 정의되었고, $\nu_n = \nu_0 + n$ , $\boldsymbol{S_n} = \boldsymbol{S_0} + \boldsymbol{S_{\theta}}$입니다.

이 두 가지 full conditional distribution은 깁스 샘플러(Gibbs sampler)를 구축하는데 사용됩니다. 그리고 깁스 샘플러는 MCMC(Markov Chain Monte Carlo)를 통해 결합 사후 분포 $p(\boldsymbol{\theta}, \Sigma | \mathbf{y_1}, ..., \mathbf{y_n})$로 근사시킵니다.

시작 지점의 값 $\Sigma^{(0)}$이 주어졌을 때, 깁스 샘플러는 다음과 같은 두 단계를 통해 {$\theta^{(s)}, \Sigma^{(s)}$}에서 {$\theta^{(s+1)}, \Sigma^{(s+1)}$}을 만들 수 있습니다.

1. full conditional distribution에서 $\boldsymbol{\theta}^{(s+1)}$ 뽑기

    a) $\mathbf{y_1}, ..., \mathbf{y_n}$와 $\Sigma^{(s)}$로 부터 $\boldsymbol{\mu_n}$과 $\Lambda_n$을 계산한다
    
    b) MVN($\mathbf{\mu_n}, \Lambda_n$)에서 $\mathbf{\theta}^{(s+1)}$을 뽑는다

2. full conditional distribution에서 $\Sigma^{s+1}$ 뽑기

    a) $\mathbf{y_1}, ..., \mathbf{y_n}$와 위에서 뽑은 $\boldsymbol{\theta}^{(s+1)}$로 부터 $\mathbf{S_n}$ 계산하기
    
    b) inverse-Wishart($\nu_0 + n, \boldsymbol{S_n}^{-1}$)에서 $\Sigma^{(s+1)}$ 뽑기

1-a)단계와 2-a)단계에서 중요한 사실은 {$\mu_n, \Lambda_n$}이 $\Sigma$값에 의존하고, $\boldsymbol{S_n}$이 $\mathbf{\theta}$에 의존하기 때문에 깁스 샘플러의 각각의 반복 회차에서 이 값들을 다시 계산해야만 한다는 것입니다.

### 예제 : 독해 시험

사전 정보 : 

22명의 학생들이 1차, 2차 시험을 보고 그 22쌍의 성적들은 i.i.d. 다변량 정규분포를 따른다고 가정한다. 그리고 그 평균 점수는 50~100 사이의 값이 나오도록 설정되었습니다.

목표 : 

사후 기댓값 $\mathbf{\theta}$와 사후 분산 $\Sigma$ 추정

사전 파라미터 설정 :

1. 사전 기댓값

$\mu_0 = (50,50)^T \Rightarrow$ 평균 점수가 50점에서 100점 사이이므로

2. 사전 공분산 행렬($\Lambda_0$)

점수가 0부터 100 사이의 값이 나와야 하므로, 이 구간 밖의 값이 나올 확률은 최대한 낮게 설정해야되기 때문에

$\theta_1$의 사전 분산($= \lambda_{0,1}^2$) = $\theta_2$의 사전 분산($= \lambda_{0,2}^2$) = $(50/2)^2$ = 625

로 설정하면 $Pr(\theta_j \notin [0,100]) = 0.05$가 됩니다.

사전 공분산 행렬 $\Sigma$의 사전 분포 또한 이러한 시험 성적이 가질 수 있는 값에 대한 범위에 대한 몇 가지 로직이 적용됩니다.

두 개의 시험이 비슷한 것을 측정하기 때문에, $\theta_1, \theta_2$의 값이 어떻든 비슷한 값을 가질 확률이 높습니다. 

따라서 이것을 사전 상관관계를 0.5로 설정하면서 반영하겠습니다. 즉 $\lambda_{1,2} = 312.5$입니다.

4. 사전 표본 크기($\nu_0$와 잔차제곱합 행렬($\boldsymbol{S_0}$)

$\boldsymbol{S_0}$은 $\Lambda_0$과 같다고 설정합니다.

하지만 $\nu_0 = p+2 = 4$로 설정함으로써 $\Sigma$를 중심으로 느슨하게 퍼져있다고 하겠습니다.

즉 다음과 같이 설정하도록 합시다.


```python
import numpy as np

mu0 = np.array([50,50])
L0 = np.array([[625, 312.5], [312.5,625]])

nu0 = 4
S0 = np.array([[625, 312.5], [312.5,625]])
```

데이터 : 

관찰된 데이터 값 $\mathbf{y_1}, ..., \mathbf{y_2}$들은 **그림 6.2**의 두 번째 그래프에 점으로 찍혀있습니다.

표본 평균

$\mathbf{\bar{y}} = (47.18, 53.86)^T$

표본 분산

$s_1^2 = 182.16$

$s_2^2 = 243.65$

표본 상관관계

$s_{1,2}/(s_1 s_2) = 0.70$


```python
import pandas as pd

Y = np.array([[59, 43, 34, 32, 42, 38, 55, 67, 64, 45, 49, 72, 34, 
70, 34, 50, 41, 52, 60, 34, 28, 35],[77, 39, 46, 26, 38, 43, 68, 
86, 77, 60, 50, 59, 38, 48, 55, 58, 54, 60, 75, 47, 48, 33]]).transpose()

Y = pd.DataFrame(Y)
Y.columns = ['firsttest', 'secondtest']
```

자 이제 깁스샘플러를 사용하여 이 데이터와 미리 설정한 사전 분포를 결합해 모수들의 추정값과 신뢰구간을 구해보도록 하겠습니다. 

우선 $\Sigma^{(0)}$을 표본 공분산 행렬과 같다고 설정하고 그것으로 부터 반복을 해보겠습니다.


```python
n = data.shape[0]
Sigma = Y.cov()
ybar = Y.apply(np.mean, axis = 0)

import random
from scipy.stats import wishart



random.seed(1)
THETA = pd.DataFrame()
SIGMA = list()

for s in np.arange(0,5000):
    
    # theta 업데이트
    
    Ln = np.linalg.inv( np.linalg.inv(L0) + n * np.linalg.inv(Sigma) )
    mun = np.dot(Ln, np.dot(np.linalg.inv(L0), mu0) + n * np.dot(np.linalg.inv(Sigma), ybar))
    theta = np.random.multivariate_normal(mun, Ln, 1)
    
    
    # Sigma 업데이트
    
    Sn = S0 + np.dot(np.transpose(Y - theta), Y-theta)
    Sigma = np.linalg.inv(wishart.rvs(nu0+n, np.linalg.inv(Sn)))
    
    # 결과 저장
    
    THETA = THETA.append(pd.DataFrame(theta))
    SIGMA.append(Sigma.tolist())
    
    
```

위의 코드로부터 경험적 분포가 $p(\boldsymbol{\theta}, \Sigma|\mathbf{y_1}, ..., \mathbf{y_n})$으로 근사하는 5000개의 값 ({$\boldsymbol{\theta}^{(1)}, \Sigma^{(1)}$}, ..., {$\boldsymbol{\theta}^{(5000)}, \Sigma^{(5000)}$})들을 생성했습니다(수렴성과 자기상관성 검사는 exercise로 남겨놓겠습니다!). 이 표본들로부터 사후 확률과 신뢰구간을 근사할 수 있게 됩니다.


```python
pd.DataFrame(THETA[1] - THETA[0]).quantile(q = [0.025, 0.5, 0.975]).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.025</th>
      <th>0.500</th>
      <th>0.975</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.346646</td>
      <td>6.54119</td>
      <td>11.440845</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.mean(THETA[1] > THETA[0])
```




    0.9926



사후 확률 $Pr(\theta_2 > \theta_1 | \mathbf{y_1}, ..., \mathbf{y_n}) = 0.99$가 의미하는 것은 우리가 큰 모집단에서 시험과 교육을 진행한다면 교육을 진행한 후 치러진 2차 시험의 평균 성적이 1차 시험보다 더 높다는 강한 증거입니다.

이것의 증거는 **그림 6.2**의 첫 번째 그래프에서 시각적으로 볼 수 있습니다. 이 그래프는 $\boldsymbol{\theta} = (\theta_1, \theta_2)^T$의 결합 사후 분포에서 97.5%, 75%, 50%, 25% 그리고 2.5% HPD(Highest Posterior Density) 등고선을 보여줍니다. HPD 등고선은 2차원에서의 신뢰구간과 유사한 개념입니다. $\boldsymbol{\theta}$의 사후 분포 등고선은 대부분이 45도 선 $\theta_1 = \theta_2$ 위에 위치해있는 것을 확인할 수 있습니다.

![IMG_E870DBFB2139-1](https://user-images.githubusercontent.com/57588650/112321458-5e2ed080-8cf3-11eb-9ad6-ea19e10d7b7b.jpeg)

**그림. 6.2** 독해 시험 데이터와 사후 분포

살짝 다른 얘기를 해봅시다. 그렇다면 무작위로 선택된 학생이 첫 번째 시험 보다 두 번째 시험을 더 잘봤을 확률은 어떻게 될까요? 그 답은 관찰값이 주어졌을 때 새로운 표본 ($Y_1, Y_2)^T$ 의 사후 예측 분포가 답이 될 수 있습니다. **그림. 6.2**에서 두 번째 그래프가 바로 사후 예측 분포의 HPD 등고선을 보여주고, 이는 $y_1 = y_2$ 겹치는 부분이 많긴 하지만 직선보다 대부분 위에 있다는 것을 알 수 있습니다.  

실제로 Pr$(Y_2 > Y_1 | \mathbf{y_1}, ..., \mathbf{y_n}) = 0.71$입니다. 그렇다면 어떻게 시험 사이에 있는 교육의 효과성을 측정할 수 있을까요? 한쪽 측면에서 보면 Pr($\theta_2 > \theta_1 | \mathbf{y_1}, ..., \mathbf{y_n}) = 0.99$이기 때문에 "높은 수준의 차이"를 보인다고 할 수도 있고 Pr$(Y_2 > Y_1 | \mathbf{y_1}, ..., \mathbf{y_n}) = 0.71$의 측면에서는 3분의 1 정도의 학생이 두 번째 시험에서 더 낮은 점수를 받을 수도 있다고 해석할 수도 있습니다. 이 두 가지 확률의 차이점은 첫 번째 확률이 $\theta_2$가 $\theta_1$보다 더 크다는 증거를 측정할 때, $\theta_2 - \theta_1$의 차이의 크기가 표본 변동성과 비교했을 때 크다는 것을 고려하지 않았기 때문입니다. 이러한 모집단을 비교하는 두 가지 방식의 혼동은 실험이나 설문조사 보고서를 작성할 대 빈번하게 일어납니다. 아주 큰 표본의 크기를 가지고 있는 연구들은 1과 아주 가까운 Pr($\theta_2 > \theta_1 | \mathbf{y_1}, ..., \mathbf{y_n})$를 결과로 산출(또는 p-value가 0에 아주 가깝다고 제안합니다)하게 되고 "큰 효과가 있다"고 제안합니다. 비록 그러한 결과가 무작위로 추출된 한 단위에 얼마나 큰 영향을 끼치는지는 알지 못하지만요.
